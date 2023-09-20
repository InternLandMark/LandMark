# pylint: disable=E1102,E1136
import os

import numpy as np
import torch
import torch.nn.functional as F

from .alpha_mask import AlphaGridMask
from .gridnerf_parallel import GridBaseParallel


class DistRenderGridNeRFTensorParallel(GridBaseParallel):
    """
    GridNeRF with tensor parallel for distributed rendering

    Args:
        aabb (torch.Tensor): Axis-aligned bounding box
        gridSize (torch.Tensor): Size of grid
        device (torch.device): Device that the model runs on.
    """

    def __init__(self, aabb, gridSize, device, **kargs):
        self.tensor_parallel_local_rank = kargs["args"].tensor_parallel_local_rank
        self.tensor_parallel_group_world_size = kargs["args"].tensor_parallel_group_world_size
        self.tensor_parallel_group = kargs["args"].tensor_parallel_group
        self.tensor_parallel_all_gather_func = kargs["args"].tensor_parallel_all_gather_func
        self.data_parallel2_scatter_func = kargs["args"].data_parallel2_scatter_func
        self.half_precision_param = kargs["args"].half_precision_param

        super().__init__(aabb, gridSize, device, **kargs)

    def update_alpha_mask(self, ckpt):
        if "alphaMask.aabb" in ckpt.keys():
            length = np.prod(ckpt["alphaMask.shape"])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt["alphaMask.mask"])[:length].reshape(ckpt["alphaMask.shape"])
            )

            self.alphaMask = AlphaGridMask(
                self.device,
                ckpt["alphaMask.aabb"].to(self.device),
                alpha_volume.float().to(self.device),
            )

    def load(self, ckpt):
        """load checkpoint

        Args:
            ckpt (dict): checkpoint
        """
        self.update_alpha_mask(ckpt)
        self.load_state_dict(ckpt["state_dict"], strict=False)

    def init_svd_volume(self, device):
        """
        Init svd volume parameters

        Args:
            device (torch.device): Device that the model runs on
        """
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)

        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp) * len(self.resMode), self.app_dim, bias=False, device=device
        )

        if self.nonlinear_density:
            self.basis_den = torch.nn.Linear(sum(self.density_n_comp) * len(self.resMode), 1, bias=False, device=device)
        if self.args.encode_app:
            self.embedding_app = torch.nn.Embedding(11500, 48, device=device)
        print(self.density_plane)

    def init_per_comp(self, plane_coef, line_coef, per_comp_size, planeSize, lineSize, scale):
        if self.half_precision_param:
            split_plane_chunk = torch.randn((1, per_comp_size, *planeSize), dtype=torch.half)
        else:
            split_plane_chunk = torch.randn((1, per_comp_size, *planeSize))
        plane_coef.append(torch.nn.Parameter(scale * split_plane_chunk))
        split_line_chunk = torch.randn((1, per_comp_size, *lineSize))
        line_coef.append(torch.nn.Parameter(scale * split_line_chunk))

    def init_one_svd(self, n_component, gridSize, scale, device):  # pylint: disable=W0613
        """
        Init one svd (feature plane or feature line)

        Args:
            n_component (int): number of components of volume grid
            gridSize (torch.Tensor): the xyz size of volume grid
            scale (float): scale of initial parameters
            device (torch.device): Device that the model runs on

        Returns:
            torch.nn.ParameterList: parameters of volume grid
        """

        plane_coef, line_coef = [], []
        # set gridsize according to ckpt, since the scale relationship between different planes is broken
        # when using merged ckpts trained by plane parallel
        ckpt_fp = self.args.ckpt[:-3] + "-wo_state_dict.th"
        if not os.path.exists(ckpt_fp):
            print("Warning: there is no -wo_state_dict.th file, the loading will be slow.")
            ckpt = torch.load(self.args.ckpt, map_location="cpu")
            for i in range(len(self.vecMode)):
                assert n_component[i] % self.tensor_parallel_group_world_size == 0
                per_comp_size = int(n_component[i] / self.tensor_parallel_group_world_size)
                for j in range(len(self.resMode)):
                    # just need to use density plane to get both density/app plane size
                    self.plane_dim = 3 if ckpt["state_dict"][f"density_plane.{j}"].dim() == 5 else 2
                    planeSize = ckpt["state_dict"][f"density_plane.{j}"].shape[-self.plane_dim :]
                    lineSize = ckpt["state_dict"][f"density_line.{j}"].shape[-2:]
                    self.init_per_comp(plane_coef, line_coef, per_comp_size, planeSize, lineSize, scale)
            del ckpt
        else:
            ckpt = torch.load(ckpt_fp, map_location="cpu")
            for i in range(len(self.vecMode)):
                assert n_component[i] % self.tensor_parallel_group_world_size == 0
                per_comp_size = int(n_component[i] / self.tensor_parallel_group_world_size)
                for j in range(len(self.resMode)):
                    self.plane_dim = 3 if len(ckpt["gridShape"][f"density_plane.{j}"]) == 5 else 2
                    planeSize = ckpt["gridShape"][f"density_plane.{j}"][-self.plane_dim :]
                    lineSize = ckpt["gridShape"][f"density_line.{j}"][-2:]
                    self.init_per_comp(plane_coef, line_coef, per_comp_size, planeSize, lineSize, scale)
            del ckpt
        assert len(plane_coef) > 0
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    # optimization
    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        """get model parameters and learning rate

        Args:
            lr_init_spatial (float, optional): Initial learning rate of gridbased parameters. Defaults to 0.02.
            lr_init_network (float, optional): Initial learning rate of MLP based parameters. Defaults to 0.001.

        Returns:
            dict: model parameters and learning rate
        """
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatial},
            {"params": self.density_plane, "lr": lr_init_spatial},
            {"params": self.app_line, "lr": lr_init_spatial},
            {"params": self.app_plane, "lr": lr_init_spatial},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if self.args.encode_app:
            grad_vars += [{"params": self.embedding_app.parameters(), "lr": lr_init_network}]
        if self.nonlinear_density:
            grad_vars += [{"params": self.basis_den.parameters(), "lr": lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{"params": self.renderModule.parameters(), "lr": lr_init_network}]
        if self.run_nerf:
            grad_vars += [{"params": self.nerf.parameters(), "lr": 5e-4}]
        return grad_vars

    # obtain feature grids
    def compute_densityfeature(self, xyz_sampled, return_feat=False, use_xyzb=False):
        """compute density feature of GridNeRF

        Args:
            xyz_sampled (torch.Tensor): xyz of sampled points
            return_feat (bool, optional): whether return density feature. Defaults to False.
            use_xyzb (bool, optional): will not be used in channel parallel. Defaults to False.

        Returns:
            torch.Tensor: density feature of GridNeRF
        """
        N = self.ndims
        if self.plane_dim == 3:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., [*self.matMode[i], self.matMode[i][1] + 2]] for i in range(N)])
                .detach()
                .view(N, 1, -1, 1, 3)
            )
        else:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
            )

        if self.half_precision_param:
            coordinate_plane = coordinate_plane.half()

        if use_xyzb:
            coordinate_line = (
                torch.stack([xyz_sampled[..., [self.vecMode[i] + 1, self.vecMode[i]]] for i in range(N)])
                .detach()
                .view(N, -1, 1, 2)
            )
        else:
            coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
            coordinate_line = (
                torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
            )
        # either direct summation over interpolation / mlp mapping
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = [], []

        single_rank_coefs = []
        for idx_plane in range(len(self.density_plane)):
            idx_dim = idx_plane // len(self.resMode)
            plane_coef = F.grid_sample(
                self.density_plane[idx_plane],
                coordinate_plane[[idx_dim]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            line_coef = F.grid_sample(
                self.density_line[idx_plane], coordinate_line[[idx_dim]], align_corners=True
            ).view(-1, *xyz_sampled.shape[:1])
            if self.nonlinear_density or return_feat:
                plane_coef_point.append(plane_coef)
                line_coef_point.append(line_coef)

            single_rank_coefs.append(plane_coef * line_coef)

        single_rank_coef = torch.cat(single_rank_coefs)
        all_rank_coef = self.tensor_parallel_all_gather_func(single_rank_coef)
        sigma_feature = torch.sum(all_rank_coef, dim=0)

        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        if self.nonlinear_density:
            sigma_feature = F.relu(self.basis_den((plane_coef_point * line_coef_point).T))[..., 0]

        if return_feat:
            ret = (plane_coef_point * line_coef_point).T
            return sigma_feature, ret

        return sigma_feature

    def compute_appfeature(self, xyz_sampled, return_feat=False, use_xyzb=False):
        """compute appearance feature of GridNeRF

        Args:
            xyz_sampled (torch.Tensor): xyz of sampled points
            return_feat (bool, optional): whether return density feature. Defaults to False.
            use_xyzb (bool, optional): will not be used in channel parallel. Defaults to False.

        Returns:
            torch.Tensor: appearance feature of GridNeRF
        """
        N = self.ndims
        if self.plane_dim == 3:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., [*self.matMode[i], self.matMode[i][1] + 2]] for i in range(N)])
                .detach()
                .view(N, 1, -1, 1, 3)
            )
        else:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
            )

        if self.half_precision_param:
            coordinate_plane = coordinate_plane.half()

        if use_xyzb:
            coordinate_line = (
                torch.stack([xyz_sampled[..., [self.vecMode[i] + 1, self.vecMode[i]]] for i in range(N)])
                .detach()
                .view(N, -1, 1, 2)
            )
        else:
            coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
            coordinate_line = (
                torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
            )
        plane_coef_point, line_coef_point = [], []

        for idx_plane in range(len(self.app_plane)):
            idx_dim = idx_plane // len(self.resMode)
            plane_coef_point.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    coordinate_plane[[idx_dim]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:-1])
            )
            line_coef_point.append(
                F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_dim]], align_corners=True).view(
                    -1, *xyz_sampled.shape[:1]
                )
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        single_rank_coef_point = plane_coef_point * line_coef_point
        all_rank_coef_point = self.tensor_parallel_all_gather_func(single_rank_coef_point)
        all_rank_coef_point = (
            all_rank_coef_point.reshape(
                self.tensor_parallel_group_world_size, len(self.app_plane), -1, all_rank_coef_point.shape[-1]
            )
            .transpose(0, 1)
            .flatten(0, 2)
        )

        full_mat_input = (all_rank_coef_point).T
        single_rank_mat_input, _ = self.data_parallel2_scatter_func(full_mat_input)
        ret = self.basis_mat(single_rank_mat_input)

        return ret

    def compute_app_latent(self, xyzb_sampled, app_mask, app_code, xyz_sampled_idxs):
        if self.args.encode_app:
            if xyz_sampled_idxs is None:
                fake_xyzb_sampled_idxs = (
                    torch.ones(xyzb_sampled.shape[:-1], dtype=torch.long, device=self.device) * app_code.long()
                )
                single, _ = self.data_parallel2_scatter_func(fake_xyzb_sampled_idxs[app_mask])
                app_latent = self.embedding_app(single)
            else:
                raise NotImplementedError()
        else:
            app_latent = None
        return app_latent

    def compute_app_render(self, xyz_sampled, app_mask, viewdirs, rgb, use_xyzb, app_code=2930, xyz_sampled_idxs=None):
        single_app_features = self.compute_appfeature(xyz_sampled[app_mask], use_xyzb=use_xyzb)
        single_app_latent = self.compute_app_latent(
            xyz_sampled, app_mask, app_code=app_code, xyz_sampled_idxs=xyz_sampled_idxs
        )
        single_app_viewdirs, gap = self.data_parallel2_scatter_func(viewdirs[app_mask])
        valid_rgbs = self.renderModule(
            single_app_viewdirs,
            single_app_features,
            single_app_latent,
        )
        full_valid_rgbs = self.tensor_parallel_all_gather_func(valid_rgbs)
        if gap > 0:
            full_valid_rgbs = torch.cat((full_valid_rgbs, full_valid_rgbs[-gap:]), 0)
        if full_valid_rgbs.shape[0] > 0:
            rgb[app_mask] = full_valid_rgbs
        return rgb, None, None

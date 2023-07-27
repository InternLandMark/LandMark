# pylint: disable=E1136
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed import ReduceOp

from .alpha_mask import AlphaGridMask
from .comm import AllGather, AllReduce
from .gridnerf_parallel import GridBaseParallel


class GridNeRFChannelParallel(GridBaseParallel):
    """
    GridNeRF with channel parallel

    Args:
        aabb (torch.Tensor): Axis-aligned bounding box
        gridSize (torch.Tensor): Size of grid
        device (torch.device): Device that the model runs on.
    """

    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, device):
        """ "
        Init svd volume parameters

        Args:
            device (torch.device): Device that the model runs on
        """
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp) * len(self.resMode), self.app_dim, bias=False).to(device)
        if self.nonlinear_density:
            self.basis_den = torch.nn.Linear(sum(self.density_n_comp) * len(self.resMode), 1, bias=False).to(device)

        print("density plane: ", self.density_plane)

    def init_one_svd(self, n_component, gridSize, scale, device):
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
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            for j in self.resMode:
                plane_coef.append(
                    torch.nn.Parameter(
                        scale
                        * torch.chunk(
                            torch.randn(
                                (
                                    1,
                                    n_component[i],
                                    gridSize[mat_id_1] * j,
                                    gridSize[mat_id_0] * j,
                                )
                            ),
                            self.world_size,
                            dim=1,
                        )[self.rank]
                    )
                )  #
                line_coef.append(
                    torch.nn.Parameter(
                        scale
                        * torch.chunk(
                            torch.randn((1, n_component[i], gridSize[vec_id] * j, 1)),
                            self.world_size,
                            dim=1,
                        )[self.rank]
                    )
                )
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def all_gather_func(self, tensor):
        """wrapped all_gather function

        Args:
            tensor (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: garhered tensor
        """
        return AllGather.apply(tensor, self.rank, self.world_size, self.group)

    # wrapped all_reduce function
    def all_reduce_func(self, tensor):
        tensor.div_(self.world_size)
        return AllReduce.apply(ReduceOp.SUM, self.group, tensor)

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
        if self.nonlinear_density:
            grad_vars += [{"params": self.basis_den.parameters(), "lr": lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{"params": self.renderModule.parameters(), "lr": lr_init_network}]
        if self.run_nerf:
            grad_vars += [{"params": self.nerf.parameters(), "lr": 5e-4}]
        return grad_vars

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
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
        coordinate_line = (
            torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
        )
        # either direct summation over interpolation / mlp mapping
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = [], []
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        for idx_plane in range(len(self.density_plane)):
            idx_dim = idx_plane // len(self.resMode)
            # use all_gather to get feature with full channels
            plane_coef = self.all_gather_func(
                F.grid_sample(
                    self.density_plane[idx_plane],
                    coordinate_plane[[idx_dim]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            line_coef = self.all_gather_func(
                F.grid_sample(
                    self.density_line[idx_plane],
                    coordinate_line[[idx_dim]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            if self.nonlinear_density or return_feat:
                plane_coef_point.append(plane_coef)
                line_coef_point.append(line_coef)
            sigma_feature = sigma_feature + torch.sum(plane_coef * line_coef, dim=0)
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        if self.nonlinear_density:
            sigma_feature = F.relu(self.basis_den.forward((plane_coef_point * line_coef_point).T))[..., 0]
        if return_feat:
            return sigma_feature, (plane_coef_point * line_coef_point).T

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
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
        coordinate_line = (
            torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
        )
        plane_coef_point, line_coef_point = [], []

        # for idx_plane in range(len(self.app_plane)):
        for idx_plane in range(len(self.app_plane)):
            idx_dim = idx_plane // len(self.resMode)
            # use all_gather to get feature with full channels
            plane_coef_point.append(
                self.all_gather_func(
                    F.grid_sample(
                        self.app_plane[idx_plane],
                        coordinate_plane[[idx_dim]],
                        align_corners=True,
                    ).view(-1, *xyz_sampled.shape[:-1])
                )
            )
            line_coef_point.append(
                self.all_gather_func(
                    F.grid_sample(
                        self.app_line[idx_plane],
                        coordinate_line[[idx_dim]],
                        align_corners=True,
                    ).view(-1, *xyz_sampled.shape[:1])
                )
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        if return_feat:
            return (
                self.basis_mat.forward((plane_coef_point * line_coef_point).T),
                (plane_coef_point * line_coef_point).T,
            )
        return self.basis_mat.forward((plane_coef_point * line_coef_point).T)

    # upsample/update grids
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        """
        Upsamples the plane and line coefficients to a target resolution.

        Args:
            plane_coef (list): The plane coefficients.
            line_coef (DDP): The line coefficients.
            res_target (list): The target resolution.

        Returns:
            tuple: A tuple containing the upsampled plane and line coefficients.
        """
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            for j in self.resMode:
                plane_idx = i * len(self.resMode) + self.resMode.index(j)
                plane_coef[plane_idx] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef[plane_idx].data,
                        size=(res_target[mat_id_1] * j, res_target[mat_id_0] * j),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
                line_coef[plane_idx] = torch.nn.Parameter(
                    F.interpolate(
                        line_coef[plane_idx].data,
                        size=(res_target[vec_id] * j, 1),
                        mode="bilinear",
                        align_corners=True,
                    )
                )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        """
        Upsamples the volume grid to a target resolution.

        Args:
            res_target (list): The target resolution.
        """
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)
        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")

    def vectorDiffs(self, vector_comps):
        """
        Computes the mean absolute difference between non-diagonal elements of the dot product of the input vectors.
        In channel parallel, vector with partial components will be all_gathered before matmul.

        Args:
            vector_comps (list): A list of vectors.

        Returns:
            torch.Tensor: The mean absolute difference between non-diagonal elements of the dot product of \
                the input vectors.
        """
        total = 0
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            vcmat = self.all_gather_func(vector_comps[idx].view(n_comp, n_size))
            dotp = torch.matmul(vcmat, vcmat.transpose(-1, -2))
            n_comp = n_comp * self.world_size
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        """
        Computes the sum of the mean absolute difference between non-diagonal elements of the dot product of \
                the density and appearance line coefficients.

        Returns:
            torch.Tensor: The sum of the mean absolute difference between non-diagonal elements of the dot \
                product of the density and appearance line coefficients.
        """
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        """
        Computes the L1 loss for density plane and density line
        In channel parallel, we need to do all_gather on all ranks.

        Returns:
            torch.Tensor: The loss value
        """
        total = 0
        for idx in range(len(self.density_plane)):
            dp = self.all_gather_func(torch.mean(torch.abs(self.density_plane[idx])).view([1, 1]))
            dl = self.all_gather_func(torch.mean(torch.abs(self.density_line[idx])).view([1, 1]))
            total = total + torch.mean(dp) + torch.mean(dl)
        return total

    def TV_loss_density(self, reg):
        """
        Computes the total variation loss for the density plane coefficients.
        In channel parallel, we need to do all_gather on all ranks.

        Args:
            reg (function): The regularization function.

        Returns:
            torch.Tensor: The total variation loss for the density plane coefficients.
        """
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2
        return torch.mean(self.all_gather_func(total.view([1, 1])))

    def TV_loss_app(self, reg):
        """
        Computes the total variation loss for the appearance plane coefficients.
        In channel parallel, we need to do all_gather on all ranks.

        Args:
            reg (function): The regularization function.

        Returns:
            torch.Tensor: The total variation loss for the appearance plane coefficients.
        """
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2
        return torch.mean(self.all_gather_func(total.view([1, 1])))

    def load(self, ckpt):
        """load checkpoint and split at the dim of channels

        Args:
            ckpt (str): the path of checkpoint
        """
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
        split_module_names = ["density_plane", "density_line", "app_plane", "app_line"]
        keys = sorted(ckpt["state_dict"].keys())
        for key in keys:
            flag = False
            for name in split_module_names:
                if name in key:
                    flag = True
            if flag:
                split_tensor = torch.chunk(ckpt["state_dict"][key], self.world_size, dim=1)[self.rank]
                ckpt["state_dict"][key] = split_tensor
        self.load_state_dict(ckpt["state_dict"], strict=False)

    def merge_ckpts(self, logfolder):
        """
        Merge all the checkpoints into one, so that it can be used to render on one device.

        Args:
            logfolder: the folder to load the sub checkpoints and save the merged checkpoint
        """
        ckpts = []
        merged_ckpt = {}
        for rank in range(self.args.world_size):
            ckpt = f"{logfolder}/{self.args.expname}-sub{rank}.th"
            print(ckpt)
            ckpts.append(torch.load(ckpt, map_location=self.args.device))

        # set state_dict
        import copy

        merged_ckpt = copy.deepcopy(ckpts[0])
        merged_module_names = ["density_plane", "density_line", "app_plane", "app_line"]
        keys = sorted(merged_ckpt["state_dict"].keys())
        for key in keys:
            flag = False
            for name in merged_module_names:
                if name in key:
                    flag = True
            if flag:
                _, n_channel, h, w = merged_ckpt["state_dict"][key].shape
                global_shape = (1, n_channel * self.args.world_size, h, w)
                merged_tensor = torch.zeros(global_shape, device=self.device)
                for rank in range(self.args.world_size):
                    merged_tensor[0, n_channel * rank : n_channel * (rank + 1), ...] = ckpts[rank]["state_dict"][key]
                merged_ckpt["state_dict"][key] = merged_tensor

        new_gridSize = [
            merged_ckpt["state_dict"]["density_plane.0"].shape[-1],
            merged_ckpt["state_dict"]["density_plane.0"].shape[-2],
            merged_ckpt["state_dict"]["density_line.0"].shape[-2],
        ]
        merged_ckpt["kwargs"].update({"gridSize": new_gridSize})
        kwargs = ckpts[0]["kwargs"]
        kwargs.update({"device": self.args.device})
        merged_ckpt_fp = f"{logfolder}/{self.args.expname}-merged.th"
        torch.save(merged_ckpt, merged_ckpt_fp)

import copy
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from app.tools.slurm import get_dp_group
from app.tools.utils import rm_redundant_words_in_state_dict

from .comm import AllGather
from .gridnerf_parallel import GridBaseParallel, WrapParam


class GridNeRFBranchParallel(GridBaseParallel):
    """
    Branch parallel version of GridNeRF
    """

    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, device):
        """
        Initialize the SVD volume for density and apperance.

        Args:
            device (torch.device): The device to use.
        """
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        if self.args.DDP and self.is_train:
            ddp_group = get_dp_group()
            self.density_plane = DDP(self.density_plane, device_ids=[self.args.local_rank], process_group=ddp_group)
            self.density_line = DDP(self.density_line, device_ids=[self.args.local_rank], process_group=ddp_group)
            self.app_plane = DDP(self.app_plane, device_ids=[self.args.local_rank], process_group=ddp_group)
            self.app_line = DDP(self.app_line, device_ids=[self.args.local_rank], process_group=ddp_group)

        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp) * len(self.resMode), self.app_dim, bias=False, device=device
        )
        if self.is_train:
            self.basis_mat = DDP(self.basis_mat, device_ids=[self.args.local_rank], process_group=None)
        if self.nonlinear_density:
            self.basis_den = torch.nn.Linear(sum(self.density_n_comp) * len(self.resMode), 1, bias=False, device=device)
            if self.is_train:
                self.basis_den = DDP(self.basis_den, device_ids=[self.args.local_rank], process_group=None)

        if self.args.encode_app:
            self.embedding_app = torch.nn.Embedding(
                11500, 48, device=device
            )  # (N, ch), N: total num of imgs in dataset
            if self.is_train:
                self.embedding_app = DDP(self.embedding_app, device_ids=[self.args.local_rank], process_group=None)

    def init_one_svd(self, n_component, gridSize, scale, device):
        """
        Initialize one SVD volume for a density or appearance.

        Args:
            n_component (int): The number of components.
            gridSize (list): The size of the grid.
            scale (float): The scaling factor.
            device (torch.device): The device to use for computation.

        Returns:
            tuple: A tuple containing the plane and line coefficients.
        """
        plane_coef, line_coef = [], []
        if self.args.ckpt is None:
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                for j in self.resMode:
                    # plane_coef
                    # init same global plane and extract local plane for each rank
                    global_plane = scale * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1] * j, gridSize[mat_id_0] * j),
                        device=self.device,
                    )
                    dist.broadcast(global_plane, src=0)
                    local_shape = [
                        gridSize[mat_id_1] * j // self.args.plane_division[mat_id_1],
                        gridSize[mat_id_0] * j // self.args.plane_division[mat_id_0],
                    ]
                    x = local_shape[1] * (self.args.part // self.args.plane_division[1])
                    y = local_shape[0] * (self.args.part % self.args.plane_division[1])
                    local_plane = global_plane[..., y : y + local_shape[0], x : x + local_shape[1]].clone()
                    del global_plane
                    plane_coef.append(torch.nn.Parameter(local_plane))
                    # line_coef
                    line_coef.append(
                        torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id] * j, 1)))
                    )
        else:
            # set gridsize according to ckpt, since the scale relationship between different planes is broken
            # when using merged ckpts trained by plane parallel
            ckpt = torch.load(self.args.ckpt, map_location="cpu")
            rm_redundant_words_in_state_dict(ckpt["state_dict"], [".module", ".params"])
            for i in range(len(self.vecMode)):
                for j in range(len(self.resMode)):
                    self.plane_dim = 3 if ckpt["state_dict"][f"density_plane.{j}"].dim() == 5 else 2
                    planeSize = ckpt["state_dict"][f"density_plane.{j}"].shape[-self.plane_dim :]
                    lineSize = ckpt["state_dict"][f"density_line.{j}"].shape[-2:]
                    plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], *planeSize))))
                    line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], *lineSize))))
            del ckpt

        if self.is_train and self.args.DDP:
            return WrapParam(plane_coef).to(device), WrapParam(line_coef).to(device)

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def all_gather_func(self, tensor):
        """wrapped all_gather function

        Args:
            tensor (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: garhered tensor
        """
        return AllGather.apply(tensor, self.args.part, self.args.model_parallel_degree, self.group)

    # optimization
    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        """
        Returns the parameter groups for optimization.

        Args:
            lr_init_spatial (float): The initial learning rate for spatial parameters.
            lr_init_network (float): The initial learning rate for network parameters.

        Returns:
            list: A list of parameter groups for optimization.
        """
        grad_vars = [
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.density_plane, DDP):
            grad_vars += [
                {"params": self.density_line.parameters(), "lr": lr_init_spatial},
                {"params": self.density_plane.parameters(), "lr": lr_init_spatial},
                {"params": self.app_line.parameters(), "lr": lr_init_spatial},
                {"params": self.app_plane.parameters(), "lr": lr_init_spatial},
            ]
        else:
            grad_vars += [
                {"params": self.density_line, "lr": lr_init_spatial},
                {"params": self.density_plane, "lr": lr_init_spatial},
                {"params": self.app_line, "lr": lr_init_spatial},
                {"params": self.app_plane, "lr": lr_init_spatial},
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
        """
        Computes the density feature for a given set of sampled points.
        If use_xyzb is True, the argument xyz_sampled is actually xyzb_sampled, whose element is xyzb,
        and the b represent the block index.

        Args:
            xyz_sampled (torch.Tensor): The sampled points.
            return_feat (bool): Whether to return the feature.
            use_xyzb (bool): Whether to use xyzb coordinate.

        Returns:
            torch.Tensor: The density feature.
        """
        N = self.ndims
        if use_xyzb and self.plane_dim == 3:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., [*self.matMode[i], self.matMode[i][1] + 2]] for i in range(N)])
                .detach()
                .view(N, 1, -1, 1, 3)
            )
            coordinate_line = (
                torch.stack([xyz_sampled[..., [self.vecMode[i] + 1, self.vecMode[i]]] for i in range(N)])
                .detach()
                .view(N, -1, 1, 2)
            )
        else:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
            )
            coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
            coordinate_line = (
                torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
            )
        # either direct summation over interpolation / mlp mapping
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = [], []
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        idx_plane_rng = (
            range(len(self.density_plane.module))
            if isinstance(self.density_plane, DDP)
            else range(len(self.density_plane))
        )
        for idx_plane in idx_plane_rng:
            idx_dim = idx_plane // len(self.resMode)
            if isinstance(self.density_plane, DDP):
                plane_coef = self.density_plane(coordinate_plane, idx_plane, idx_dim).view(-1, *xyz_sampled.shape[:1])
                line_coef = self.density_line(coordinate_line, idx_plane, idx_dim).view(-1, *xyz_sampled.shape[:1])
            else:
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
            sigma_feature = sigma_feature + torch.sum(plane_coef * line_coef, dim=0)
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        if self.nonlinear_density:
            sigma_feature = F.relu(self.basis_den((plane_coef_point * line_coef_point).T))[..., 0]

        if return_feat:
            return sigma_feature, (plane_coef_point * line_coef_point).T

        return sigma_feature

    def compute_appfeature(self, xyz_sampled, return_feat=False, use_xyzb=False):
        """
        Computes the apperance feature for a given set of sampled points.
        If use_xyzb is True, the argument xyz_sampled is actually xyzb_sampled, whose element is xyzb,
        and the b represent the block index.

        Args:
            xyz_sampled (torch.Tensor): The sampled points.
            return_feat (bool): Whether to return the feature.
            use_xyzb (bool): Whether to use xyzb coordinate.

        Returns:
            torch.Tensor: The appearance feature.
        """
        N = self.ndims
        if use_xyzb and self.plane_dim == 3:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., [*self.matMode[i], self.matMode[i][1] + 2]] for i in range(N)])
                .detach()
                .view(N, 1, -1, 1, 3)
            )
            coordinate_line = (
                torch.stack([xyz_sampled[..., [self.vecMode[i] + 1, self.vecMode[i]]] for i in range(N)])
                .detach()
                .view(N, -1, 1, 2)
            )
        else:
            coordinate_plane = (
                torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
            )
            coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
            coordinate_line = (
                torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
            )
        plane_coef_point, line_coef_point = [], []

        idx_plane_rng = (
            range(len(self.app_plane.module)) if isinstance(self.app_plane, DDP) else range(len(self.app_plane))
        )
        for idx_plane in idx_plane_rng:
            idx_dim = idx_plane // len(self.resMode)
            if isinstance(self.app_plane, DDP):
                plane_coef_point.append(
                    self.app_plane(coordinate_plane, idx_plane, idx_dim).view(-1, *xyz_sampled.shape[:-1])
                )
                line_coef_point.append(
                    self.app_line(coordinate_line, idx_plane, idx_dim).view(-1, *xyz_sampled.shape[:1])
                )
            else:
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
        if return_feat:
            return (
                self.basis_mat((plane_coef_point * line_coef_point).T),
                (plane_coef_point * line_coef_point).T,
            )
        return self.basis_mat((plane_coef_point * line_coef_point).T)

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
        if isinstance(plane_coef, DDP):
            plane_coef = plane_coef.module
            line_coef = line_coef.module

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            for j in self.resMode:
                plane_idx = i * len(self.resMode) + self.resMode.index(j)
                plane_coef[plane_idx] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef[plane_idx].data,
                        size=(
                            round(res_target[mat_id_1] * j / self.args.plane_division[mat_id_1]),
                            round(res_target[mat_id_0] * j / self.args.plane_division[mat_id_0]),
                        ),
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
        if isinstance(plane_coef, WrapParam):
            ddp_group = get_dp_group()
            plane_coef = DDP(plane_coef, device_ids=[self.args.local_rank], process_group=ddp_group)
            line_coef = DDP(line_coef, device_ids=[self.args.local_rank], process_group=ddp_group)

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
        print(self.density_plane)

    def vectorDiffs(self, vector_comps):
        """
        Computes the mean absolute difference between non-diagonal elements of the dot product of the input vectors.

        Args:
            vector_comps (list): A list of vectors.

        Returns:
            torch.Tensor: The mean absolute difference between non-diagonal elements of the dot product of \
                the input vectors.
        """
        total = 0
        if isinstance(vector_comps, DDP):
            vector_comps = vector_comps.module
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            dotp = torch.matmul(
                vector_comps[idx].view(n_comp, n_size),
                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2),
            )
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
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line.line)

    def density_L1(self):
        """
        Computes the L1 loss for density plane and density line

        Returns:
            torch.Tensor: The loss value
        """
        total = 0
        if isinstance(self.density_plane, DDP):
            density_plane = self.density_plane.module
            density_line = self.density_line.module
        else:
            density_plane = self.density_plane
            density_line = self.density_line
        for idx in range(len(density_plane)):
            dp = self.all_gather_func(torch.mean(torch.abs(density_plane[idx])).view([1, 1]))
            # dl = self.all_gather_func(torch.mean(torch.abs(self.density_line[idx])).view([1, 1]))
            total = total + torch.mean(dp) / self.args.model_parallel_degree + torch.mean(torch.abs(density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        """
        Computes the total variation loss for the density plane coefficients.

        Args:
            reg (function): The regularization function.

        Returns:
            torch.Tensor: The total variation loss for the density plane coefficients.
        """
        total = 0
        if isinstance(self.density_plane, DDP):
            density_plane = self.density_plane.module
        else:
            density_plane = self.density_plane
        for idx in range(len(density_plane)):
            total = total + reg(density_plane[idx]) * 1e-2
        return torch.mean(self.all_gather_func(total.view([1, 1])))

    def TV_loss_app(self, reg):
        """
        Computes the total variation loss for the appearance plane coefficients.

        Args:
            reg (function): The regularization function.

        Returns:
            torch.Tensor: The total variation loss for the appearance plane coefficients.
        """
        total = 0
        if isinstance(self.app_plane, DDP):
            app_plane = self.app_plane.module
        else:
            app_plane = self.app_plane
        for idx in range(len(app_plane)):
            total = total + reg(app_plane[idx]) * 1e-2
        return torch.mean(self.all_gather_func(total.view([1, 1])))

    def merge_ckpts(self, logfolder):
        """
        Merge all the checkpoints into one, so that it can be used to render on one device.
        For merging plane, stack plane from all branchs to a 3D-plane (x,y, branch_idx)

        Args:
            logfolder: the folder to load the sub checkpoints and save the merged checkpoint
        """
        if self.args.rank != 0:
            return

        merged_ckpt_fp = f"{logfolder}/{self.args.expname}-merged-stack.th"
        print(f"Export merged checkpoint to {merged_ckpt_fp}")
        ckpts = []
        merged_ckpt = {}
        for part in range(self.args.model_parallel_degree):
            ckpt = f"{logfolder}/{self.args.expname}-sub{part}.th"
            ckpts.append(torch.load(ckpt, map_location="cpu"))  # load to cpu mem, which is much bigger than cuda mem.

        merged_ckpt = copy.deepcopy(ckpts[0])
        plane_names = ["density_plane", "app_plane"]
        line_names = ["density_line", "app_line"]

        gridShape = {}
        keys = sorted(merged_ckpt["state_dict"].keys())
        for key in tqdm(keys, desc="Exporting merged ckpt", file=sys.stdout):
            # merge plane
            for name in plane_names:
                if name in key:
                    _, n_channel, h, w = merged_ckpt["state_dict"][key].shape
                    global_shape = (1, n_channel, self.args.model_parallel_degree, h, w)
                    merged_tensor = torch.zeros(global_shape, device="cpu")
                    for part in range(self.args.model_parallel_degree):
                        merged_tensor[:, :, part] = ckpts[part]["state_dict"][key]
                    merged_ckpt["state_dict"][key] = merged_tensor
                    gridShape[key] = merged_tensor.shape
            for name in line_names:
                if name in key:
                    # stack line
                    global_shape = (
                        *merged_ckpt["state_dict"][key].shape[:3],
                        self.args.plane_division[0] * self.args.plane_division[1],
                    )
                    stack_tensor = torch.zeros(global_shape, device="cpu")
                    for part in range(self.args.model_parallel_degree):
                        stack_tensor[..., part] = ckpts[part]["state_dict"][key][..., 0]
                    merged_ckpt["state_dict"][key] = stack_tensor
                    gridShape[key] = stack_tensor.shape

        rm_redundant_words_in_state_dict(merged_ckpt["state_dict"], [".module", ".params"])

        new_gridSize = [
            merged_ckpt["state_dict"]["density_plane.0"].shape[-1],
            merged_ckpt["state_dict"]["density_plane.0"].shape[-2],
            merged_ckpt["state_dict"]["density_line.0"].shape[-2],
        ]
        kwargs = ckpts[0]["kwargs"]
        kwargs.update({"device": self.args.device, "args": vars(self.args), "gridSize": new_gridSize})
        merged_ckpt.update({"kwargs": kwargs})

        # save grid shape
        merged_ckpt["gridShape"] = gridShape
        torch.save(merged_ckpt, merged_ckpt_fp)

        # save ckpt without plane for dynamic load rendering
        merged_ckpt["state_dict"].pop("density_plane.0")
        merged_ckpt["state_dict"].pop("app_plane.0")
        try:
            merged_ckpt["state_dict"].pop("density_plane.1")
            merged_ckpt["state_dict"].pop("density_plane.2")
            merged_ckpt["state_dict"].pop("app_plane.1")
            merged_ckpt["state_dict"].pop("app_plane.2")
        except KeyError:
            pass

        wo_plane_ckpt_fp = merged_ckpt_fp[:-3] + "-wo_plane.th"
        torch.save(merged_ckpt, wo_plane_ckpt_fp)

        # save ckpt without state_dict, that is kwargs, for model initialization
        merged_ckpt.pop("state_dict")
        kwargs_fp = merged_ckpt_fp[:-3] + "-wo_state_dict.th"
        torch.save(merged_ckpt, kwargs_fp)


class DistRenderGridNeRFBranchParallel(GridNeRFBranchParallel):
    """
    Branch parallel GridNeRF for distributed rendering
    """

    def init_one_svd(self, n_component, gridSize, scale, device):  # pylint: disable=W0613
        """
        Initialize one SVD volume for a density or appearance.

        Args:
            n_component (int): The number of components.
            gridSize (list): The size of the grid.
            scale (float): The scaling factor.
            device (torch.device): The device to use for computation.

        Returns:
            tuple: A tuple containing the plane and line coefficients.
        """
        plane_coef, line_coef = [], []
        # set gridsize according to ckpt, since the scale relationship between different planes is broken
        # when using merged ckpts trained by plane parallel
        ckpt_fp = self.args.ckpt[:-3] + "-wo_state_dict.th"
        if not os.path.exists(ckpt_fp):
            ckpt = torch.load(self.args.ckpt, map_location="cpu")
            for i in range(len(self.vecMode)):
                for j in range(len(self.resMode)):
                    # just need to use density plane to get both density/app plane size
                    self.plane_dim = 3 if ckpt["state_dict"][f"density_plane.{j}"].dim() == 5 else 2
                    planeSize = ckpt["state_dict"][f"density_plane.{j}"].shape[-self.plane_dim :]
                    lineSize = ckpt["state_dict"][f"density_line.{j}"].shape[-2:]
                    plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], *planeSize))))
                    line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], *lineSize))))
            del ckpt
        else:
            ckpt = torch.load(ckpt_fp, map_location="cpu")
            for i in range(len(self.vecMode)):
                for j in range(len(self.resMode)):
                    self.plane_dim = 3 if len(ckpt["gridShape"][f"density_plane.{j}"]) == 5 else 2
                    planeSize = ckpt["gridShape"][f"density_plane.{j}"][-self.plane_dim :]
                    lineSize = ckpt["gridShape"][f"density_line.{j}"][-2:]
                    plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], *planeSize))))
                    line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], *lineSize))))
            del ckpt
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

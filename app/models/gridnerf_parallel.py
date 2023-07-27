# pylint: disable=E1111
import numpy as np
import torch
import torch.distributed as dist
import torch.nn
import torch.nn.functional as F
from tools.dataloader.ray_utils import sample_pdf
from tools.utils import TVLoss, raw2alpha, st
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_efficient_distloss import eff_distloss

from .alpha_mask import AlphaGridMask
from .mlp_render_fea import MLPRender_Fea
from .nerf_branch import NeRF, NeRFParallel, raw2outputs


class GridBaseParallel(torch.nn.Module):
    """
    Base class for GridNeRF with parallel

    Args:
        aabb (torch.Tensor): Axis-aligned bounding box
        gridSize (torch.Tensor): Size of grid
        device (torch.device): Device that the model runs on.
        density_n_comp (int): Number of component in the density grid.
        appearance_n_comp (int): Number of component in the appearance grid.
        app_dim (int): In channel dimension of mlp for calculating appearance.
        alphaMask (torch.nn.Module): Alpha Mask instance
        near_far (list): Near and far plane along the sample ray
        density_shift (int): Shift density in softplus
        alphaMask_thres (float): Threshold for creating alpha mask volume
        distance_scale (int): scaling sampling distance for computation
        rayMarch_weight_thres (float): Threshold for filtering weight to get the app_mask
        pos_pe (int): Number of pe for pos
        view_pe (int): Number of pe for view
        fea_pe (int): Number of pe for features
        featureC (int): Hidden feature channel in MLP
        step_ratio (int): How many grids to walk in one step during sampling
        fea2denseAct (str): Function for feature2density; only support softplus/relu
        use_plane_split (bool): Whether to split plane, usually set to False with channel parallel.
        args (ArgsConfig): Args instance that holds the config setting.
        group (torch.distributed.ProcessGroup): Distributed communication group that the model on \
            current rank belongs to.
        is_train (bool): Distinguish between training and rendering in order to init modules correctly.
    """

    def __init__(  # pylint: disable=W0102
        self,
        aabb,
        gridSize,
        device,
        density_n_comp=8,
        appearance_n_comp=24,
        app_dim=27,
        alphaMask=None,
        near_far=[2.0, 6.0],
        density_shift=-10,
        alphaMask_thres=0.001,
        distance_scale=25,
        rayMarch_weight_thres=0.001,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        featureC=128,
        step_ratio=2.0,
        fea2denseAct="softplus",
        use_plane_split=True,
        args=None,
        group=None,
        is_train=True,
    ):
        super().__init__()

        if args.distributed:
            self.rank = args.rank
            self.world_size = args.world_size
            self.group = group

        self.is_train = is_train
        self.run_nerf = args.run_nerf
        self.nonlinear_density = args.nonlinear_density
        self.ndims = args.ndims
        self.TV_weight_density = args.TV_weight_density
        self.TV_weight_app = args.TV_weight_app
        self.tvreg = TVLoss()

        self.density_n_comp = density_n_comp[: self.ndims]
        self.app_n_comp = appearance_n_comp[: self.ndims]
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]][: self.ndims]
        self.vecMode = [2, 1, 0][: self.ndims]
        self.comp_w = [1, 1, 1][: self.ndims]

        self.resMode = args.resMode if args.resMode is not None else [1]

        self.args = args

        self.sampling_opt = args.sampling_opt

        self.init_svd_volume(device)

        # feature renderer
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.fea_pe = fea_pe
        self.featureC = featureC

        self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, args.bias_enable).to(device)

        self.run_nerf = args.run_nerf
        if self.run_nerf:
            self.init_nerf(args)

        self.n_importance = args.n_importance

        self.use_plane_split = use_plane_split
        if self.use_plane_split and self.is_train:
            self.renderModule = DDP(self.renderModule, device_ids=[self.args.local_rank])
            self.register_grid_ddp_hook()

    def init_nerf(self, args):
        """
        create nerf branch
        """
        if self.use_plane_split:
            self.nerf = NeRFParallel(
                args,
                sum(self.density_n_comp) * len(self.resMode),
                sum(self.app_n_comp) * len(self.resMode),
            ).to(self.device)
            if self.is_train:
                self.nerf = DDP(self.nerf, device_ids=[self.args.local_rank])
                self.register_nerf_ddp_hook()
        else:
            self.nerf = NeRF(
                args,
                sum(self.density_n_comp) * len(self.resMode),
                sum(self.app_n_comp) * len(self.resMode),
            ).to(self.device)

        self.residnerf = args.residnerf
        self.n_importance = args.n_importance
        self.nerf_n_importance = args.nerf_n_importance
        self.run_nerf = True
        print("init run_nerf", self.nerf)

    def update_stepSize(self, gridSize):
        """
        update step size according to new grid size

        Args:
            gridSize (list): grid size
        """
        print("", flush=True)
        print(st.GREEN + "grid size" + st.RESET, gridSize, flush=True)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print(
            st.BLUE + f"sample step size: {self.stepSize:.3f} unit" + st.RESET,
            flush=True,
        )
        print(st.BLUE + f"default number: {self.nSamples}" + st.RESET, flush=True)

    def get_kwargs(self):
        return {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist(),
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "density_shift": self.density_shift,
            "alphaMask_thres": self.alphaMask_thres,
            "distance_scale": self.distance_scale,
            "rayMarch_weight_thres": self.rayMarch_weight_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
        }

    def init_svd_volume(self, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled, return_feat=False, use_xyzb=False):
        pass

    def compute_appfeature(self, xyz_sampled, return_feat=False, use_xyzb=False):
        pass

    def split_samples(self, xyz_sampled, ray_valid=None, is_train=True):
        """
        Split samples according to their x-y coordinates and the coordinate ranges of splitted-planes.

        Args:
            xyz_sampled: samples after normalized to [-1,1] by normalize_coord
            ray_valid: masks that filter the useful rays
            is_train: indicates whether during training.

        Returns:
            list: list of torch.Tensor mask that select samples belongs to the splitted-plane on this rank.
            torch.Tensor: samples that belongs to splitted-planes on current rank
        """
        masks, valid_xyzb_sampled = self.assign_blocks_to_samples(xyz_sampled, ray_valid, is_train)
        valid_xyz_sampled = valid_xyzb_sampled[..., :3]

        return masks, valid_xyz_sampled

    def assign_blocks_to_samples(self, xyz_sampled, ray_valid=None, is_train=True):
        """
        Assign samples to corresponding splitted-planes according to x-y coordinates. Append the blocks \
            index after xyz coords, compositing a 4D coords(xyzb, b:block index) Used when rendering a model \
            trained by block parallel.

        Args:
            xyz_sampled: samples after normalized to [-1,1] by normalize_coord
            ray_valid: masks that filter the useful rays
            is_train: indicates whether during training.

        Returns:
            list: list of torch.Tensor mask that select samples belongs to the splitted-plane on this rank.
            torch.Tensor: samples that belongs to splitted-planes on current rank
        """
        if ray_valid is not None:
            valid_xyz_sampled = xyz_sampled[ray_valid].clone().detach()
        else:
            valid_xyz_sampled = xyz_sampled.clone().detach()
        # generate masks
        plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]

        valid_xyzb_sampled = torch.zeros(*valid_xyz_sampled.shape[:-1], 4, device=self.device)
        valid_xyzb_sampled[..., :3] = valid_xyz_sampled

        valid_xyzb_sampled[..., :2] += 1
        valid_xyzb_sampled[..., 0] /= plane_width
        valid_xyzb_sampled[..., 1] /= plane_height
        valid_xyzb_sampled[..., 3] = torch.floor(valid_xyzb_sampled[..., 0]) * self.args.plane_division[
            1
        ] + torch.floor(valid_xyzb_sampled[..., 1])
        valid_xyzb_sampled[..., :3] = valid_xyz_sampled

        # normalized xyb coords (when using stack-merged ckpt)
        coord_min = torch.stack(
            [
                -1 + (valid_xyzb_sampled[..., 3] // self.args.plane_division[1]) * plane_width,
                -1 + (valid_xyzb_sampled[..., 3] % self.args.plane_division[1]) * plane_height,
            ],
            dim=-1,
        )
        coord_max = torch.stack(
            [
                -1 + (valid_xyzb_sampled[..., 3] // self.args.plane_division[1] + 1) * plane_width,
                -1 + (valid_xyzb_sampled[..., 3] % self.args.plane_division[1] + 1) * plane_height,
            ],
            dim=-1,
        )
        valid_xyzb_sampled[..., :2] = (valid_xyzb_sampled[..., :2] - coord_min) * 2 / (coord_max - coord_min) - 1

        masks = []
        if is_train or self.args.ckpt_type == "sub":  # no need when using full ckpt
            for block_idx in range(self.args.plane_division[0] * self.args.plane_division[1]):
                masks.append(valid_xyzb_sampled[..., 3] == block_idx)

        valid_xyzb_sampled[..., 3] = -1 + 2 * valid_xyzb_sampled[..., 3] / (
            self.args.plane_division[0] * self.args.plane_division[1] - 1
        )

        return masks, valid_xyzb_sampled

    def normalize_coord(self, xyz_sampled):
        """
        Normalize the coordinates to range [-1,1]

        Args:
            xyz_sampled: sample coordinates that will be normalized

        Returns:
            torch.Tensor: normalized xyz_sampled
        """
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def register_grid_ddp_hook(self):
        """
        A DDP communication hook function. When training with branch / plane parallel,
        it should be registered on modules that belong to grid branch.
        """

        def sigma_allreduce_hook(
            process_group: dist.ProcessGroup, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:

            batch = self.sigma_batch
            batch_sum = batch.clone()

            dist.all_reduce(batch_sum)
            if batch_sum == 0:
                weight = 0
            else:
                weight = batch / batch_sum

            # Apply the division first to avoid overflow, especially for FP16.
            tensor = bucket.buffer()
            tensor.mul_(weight)

            group_to_use = process_group if process_group is not None else dist.group.WORLD

            return (
                dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0])
            )

        def app_allreduce_hook(
            process_group: dist.ProcessGroup, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:

            batch = self.app_batch
            batch_sum = batch.clone().detach()

            dist.all_reduce(batch_sum)
            if batch_sum == 0:
                weight = 0
            else:
                weight = batch / batch_sum

            # Apply the division first to avoid overflow, especially for FP16.
            tensor = bucket.buffer()
            tensor.mul_(weight)

            group_to_use = process_group if process_group is not None else dist.group.WORLD

            return (
                dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0])
            )

        if isinstance(self.density_line, DDP) and isinstance(self.app_line, DDP):
            # if the line is wrapped by DDP, it should be registered ddp comm hook
            self.density_line.register_comm_hook(self.group, sigma_allreduce_hook)
            self.app_line.register_comm_hook(self.group, app_allreduce_hook)
        self.basis_mat.register_comm_hook(None, app_allreduce_hook)
        self.renderModule.register_comm_hook(None, app_allreduce_hook)

    def register_nerf_ddp_hook(self):
        """
        A DDP communication hook function. When training with branch / plane parallel,
        it should be registered on modules that belong to nerf branch.
        """

        def nerf_allreduce_hook(
            process_group: dist.ProcessGroup, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            batch = self.nerf_batch
            batch_sum = batch.clone().detach()

            dist.all_reduce(batch_sum)
            if batch_sum == 0:
                weight = 0
            else:
                weight = batch / batch_sum

            # Apply the division first to avoid overflow, especially for FP16.
            tensor = bucket.buffer()
            tensor.mul_(weight)

            group_to_use = process_group if process_group is not None else dist.group.WORLD

            return (
                dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0])
            )

        self.nerf.register_comm_hook(None, nerf_allreduce_hook)

    def save(self, path):
        """
        Save checkpoint that contains the kwargs, alphaMask, and model state_dict.

        Args:
            path: path to save the checkpoint
        """
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({"alphaMask.shape": alpha_volume.shape})
            ckpt.update({"alphaMask.mask": np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({"alphaMask.aabb": self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def merge_ckpts(self, logfolder):
        pass

    def load(self, ckpt):
        """
        Load checkpoint from given checkpoint path.

        Args:
            path: checkpoint path to load
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
        self.load_state_dict(ckpt["state_dict"], strict=False)

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        """
        Sample N_samples number points on a ray. If is_train is True, use segmented \
        random sampling. Otherwise use uniform sampling.

        Args:
            rays_o (torch.Tensor): origin of rays
            rays_d (torch.Tensor): direction of rays
            is_train (bool): whether in the training stage.
            N_samples (int): number of points that will be sampled on a ray.

        Returns:
            torch.Tensor: intersection points
            torch.Tensor: t values
            torch.Tensor: mask indicating if the points are inside the aabb
        """
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        aabb = self.aabb.clone()
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_within_hull(self, rays_o, rays_d, is_train=True, N_samples=-1):
        """
        Samples points along the rays within the bounding box.

        Args:
            rays_o (torch.Tensor): The origin points of the rays.
            rays_d (torch.Tensor): The direction vectors of the rays.
            is_train (bool): Whether to use random sampling or uniform sampling.
            N_samples (int): The number of samples to take along each ray.

        Returns:
            torch.Tensor: The sampled points along the rays.
            torch.Tensor: The corresponding z values.
            torch.Tensor: A boolean mask indicating whether each point is within the bounding box.
        """
        near, far = self.near_far
        o_z = rays_o[:, -1:] - self.aabb[0, 2].item()
        d_z = rays_d[:, -1:]
        far = -(o_z / d_z)
        far[rays_d[:, 2] >= 0] = self.near_far[-1]
        t_vals = torch.linspace(0.0, 1.0, steps=N_samples).to(rays_o)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
        if is_train:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(rays_o)
            z_vals = lower + (upper - lower) * t_rand

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        aabb = self.aabb
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)
        return rays_pts, z_vals, ~mask_outbbox

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        """
        Computes dense alpha values and corresponding locations on a grid.

        Args:
            gridSize (tuple): The size of the grid to use for computing the dense alpha values.

        Returns:
            torch.Tensor: The dense alpha values.
            torch.Tensor: The corresponding locations.
        """
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples
        alpha = torch.zeros_like(dense_xyz[..., 0])

        for i in range(gridSize[0]):
            alpha_pred = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize)
            alpha[i] = alpha_pred.view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200), increase_alpha_thresh=0):
        """
        Updates the alpha mask based on the current density field.

        Args:
            gridSize (tuple): The size of the grid to use for computing the alpha mask.
            increase_alpha_thresh (int): The number of orders of magnitude to increase the alpha mask threshold by.

        Returns:
            torch.Tensor: The new AABB for the alpha mask.
        """
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha >= self.alphaMask_thres * (10**increase_alpha_thresh)] = 1
        alpha[alpha < self.alphaMask_thres * (10**increase_alpha_thresh)] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb

    def feature2density(self, density_features):
        """
        Convert the given density features to a density value.

        Args:
            density_features (torch.Tensor): The density features to convert.

        Returns:
            torch.Tensor: The density value.
        """
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):
        """
        Computes the alpha values for the given locations.

        Args:
            xyz_locs (torch.Tensor): The locations to compute alpha values for.
            length (float): The length of the segment to compute alpha values for.

        Returns:
            torch.Tensor: The alpha values.
        """
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])

            if self.use_plane_split:
                # split samples
                masks, xyz_sampled = self.split_samples(xyz_sampled)

                sigma_tensor_list = [torch.zeros(mask.sum(), device=self.device) for mask in masks]
                if masks[self.args.rank].sum():
                    sigma_feature = self.compute_densityfeature(xyz_sampled[masks[self.args.rank]])
                    validsigma = self.feature2density(sigma_feature)
                else:
                    validsigma = sigma_tensor_list[self.args.rank]

                dist.all_gather(sigma_tensor_list, validsigma)
                sigma_valid = torch.zeros_like(sigma[alpha_mask], device=self.device)
                for mask, validsigma in zip(masks, sigma_tensor_list):
                    sigma_valid.masked_scatter_(mask, validsigma)

                sigma[alpha_mask] = sigma_valid  # or sigma.masked_scatter_(alpha_mask, sigma_valid)
            else:
                sigma_feature = self.compute_densityfeature(xyz_sampled)
                validsigma = self.feature2density(sigma_feature)
                sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
        return alpha

    def forward(
        self,
        rays_chunk,
        white_bg=True,
        is_train=False,
        N_samples=-1,
    ):
        rays_o = rays_chunk[:, :3]
        rays_d = viewdirs = rays_chunk[:, 3:6]

        if is_train or (not is_train and not self.sampling_opt):
            # dense sample
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_o, viewdirs, is_train=is_train, N_samples=N_samples)

            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

            if self.alphaMask is not None:
                alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
                alpha_mask = alphas > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= ~alpha_mask
                ray_valid = ~ray_invalid

            xyz_sampled = self.normalize_coord(xyz_sampled)

            if self.use_plane_split:
                if is_train or (not is_train and self.args.ckpt_type == "sub"):
                    # split samples
                    masks, valid_xyz_sampled = self.split_samples(xyz_sampled, ray_valid)
                    xyz_sampled.masked_scatter_(ray_valid.unsqueeze(-1).expand(-1, -1, 3), valid_xyz_sampled)
                else:
                    # assign samples
                    xyzb_sampled = torch.zeros(*xyz_sampled.shape[:-1], 4, device=self.device)
                    xyzb_sampled[..., :3] = xyz_sampled.clone().detach()
                    masks, valid_xyzb_sampled = self.assign_blocks_to_samples(xyz_sampled, ray_valid, is_train=True)
                    xyzb_sampled.masked_scatter_(ray_valid.unsqueeze(-1).expand(-1, -1, 4), valid_xyzb_sampled)
                    xyz_sampled = xyzb_sampled

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

            # compute sigma
            if (is_train and self.use_plane_split) or (not is_train and self.args.ckpt_type == "sub"):
                self.sigma_batch = masks[self.args.rank].sum()
                sigma_tensor_list = [
                    torch.zeros(mask.sum(), device=self.device)
                    if mask.sum() > 0 or not self.training
                    else torch.zeros((1), device=self.device)
                    for mask in masks
                ]
                if ray_valid.any() and masks[self.args.rank].any():
                    sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid][masks[self.args.rank]])
                    validsigma = self.feature2density(sigma_feature)
                else:
                    if self.training:
                        tmp_input = xyz_sampled[ray_valid][:1]
                        sigma_feature = self.compute_densityfeature(tmp_input)
                        validsigma = self.feature2density(sigma_feature)
                        sigma_tensor_list[self.args.rank] = torch.zeros((1), device=self.device)
                    else:
                        validsigma = sigma_tensor_list[self.args.rank]
                with torch.no_grad():
                    dist.all_gather(sigma_tensor_list, validsigma)
                sigma_tensor_list[self.args.rank] = validsigma
                sigma_valid = torch.zeros_like(sigma[ray_valid], device=self.device)
                for mask, validsigma in zip(masks, sigma_tensor_list):
                    sigma_valid.masked_scatter_(mask, validsigma)
                sigma.masked_scatter_(ray_valid, sigma_valid)
            else:
                if ray_valid.any():
                    sigma_feature = self.compute_densityfeature(
                        xyz_sampled[ray_valid], use_xyzb=(not is_train and self.args.ckpt_type == "full")
                    )

                    validsigma = self.feature2density(sigma_feature)
                    sigma[ray_valid] = validsigma

            _, weight, _ = raw2alpha(sigma, dists * self.distance_scale)

            app_mask = weight > self.rayMarch_weight_thres

            # compute app
            if (is_train and self.use_plane_split) or (not is_train and self.args.ckpt_type == "sub"):
                valid_app_mask = (ray_valid & app_mask)[ray_valid]
                rgb_tensor_list = [
                    torch.zeros(((mask & valid_app_mask).sum(), 3), device=self.device)
                    if (mask & valid_app_mask).sum() > 0 or not self.training
                    else torch.zeros((1, 3), device=self.device)
                    for mask in masks
                ]
                self.app_batch = (valid_app_mask & masks[self.args.rank]).sum()

                if app_mask.any() and (valid_app_mask & masks[self.args.rank]).any():
                    app_features = self.compute_appfeature(
                        xyz_sampled[ray_valid][valid_app_mask & masks[self.args.rank]]
                    )
                    valid_rgbs = self.renderModule(
                        viewdirs[ray_valid][valid_app_mask & masks[self.args.rank]],
                        app_features,
                    )

                else:
                    if self.training:  # artifical sample is only needed in training time
                        tmp_input = xyz_sampled[ray_valid][:1]
                        app_features = self.compute_appfeature(tmp_input)
                        valid_rgbs = self.renderModule(viewdirs[ray_valid][:1], app_features)
                        rgb_tensor_list[self.args.rank] = torch.zeros((1, 3), device=self.device)
                    else:
                        valid_rgbs = rgb_tensor_list[self.args.rank]

                with torch.no_grad():
                    dist.all_gather(rgb_tensor_list, valid_rgbs)
                rgb_tensor_list[self.args.rank] = valid_rgbs

                rgb_valid = torch.zeros_like(rgb[ray_valid], device=self.device)
                for mask, validrgb in zip(masks, rgb_tensor_list):
                    rgb_valid.masked_scatter_((valid_app_mask & mask).unsqueeze(-1).expand(-1, 3), validrgb)
                rgb.masked_scatter_(ray_valid.unsqueeze(-1).expand(-1, -1, 3), rgb_valid)
            else:
                if app_mask.any():
                    app_features = self.compute_appfeature(
                        xyz_sampled[app_mask], use_xyzb=(not is_train and self.args.ckpt_type == "full")
                    )
                    valid_rgbs = self.renderModule(viewdirs[app_mask], app_features)  # pylint: disable=E1102
                    rgb[app_mask] = valid_rgbs

            acc_map = torch.sum(weight, -1)
            rgb_map = torch.sum(weight[..., None] * rgb, -2)

            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1.0 - acc_map[..., None])

            rgb_map = rgb_map.clamp(0, 1)
            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, -1)

            if not is_train:
                outputs = {"rgb_map": rgb_map, "depth_map": depth_map}
            else:
                # compute distort loss
                w = weight * app_mask
                m = torch.cat(
                    (
                        (z_vals[:, 1:] + z_vals[:, :-1]) * 0.5,
                        ((z_vals[:, 1:] + z_vals[:, :-1]) * 0.5)[:, -1:],
                    ),
                    dim=-1,
                )
                dist_loss = 0.01 * eff_distloss(w, m, dists).unsqueeze(0)
                outputs = {
                    "rgb_map": rgb_map,
                    "depth_map": depth_map,
                    "distort_loss": dist_loss,
                }

        if (not is_train and self.sampling_opt) or (is_train and self.run_nerf):
            rays_o = rays_chunk[:, :3]
            rays_d = viewdirs = rays_chunk[:, 3:6]

            # importance sample for grid branch (optional)

            xyz_sampled, z_vals, ray_valid = self.sample_ray_within_hull(
                rays_o, viewdirs, is_train=is_train, N_samples=self.n_importance
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
            if self.alphaMask is not None:
                alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
                alpha_mask = alphas > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= ~alpha_mask
                ray_valid = ~ray_invalid

            xyz_sampled = self.normalize_coord(xyz_sampled)

            if self.use_plane_split:
                if is_train or (not is_train and self.args.ckpt_type == "sub"):
                    # split samples
                    masks, valid_xyz_sampled = self.split_samples(xyz_sampled, ray_valid)
                    xyz_sampled.masked_scatter_(ray_valid.unsqueeze(-1).expand(-1, -1, 3), valid_xyz_sampled)
                else:
                    # assign samples
                    xyzb_sampled = torch.zeros(*xyz_sampled.shape[:-1], 4, device=self.device)
                    xyzb_sampled[..., :3] = xyz_sampled.clone().detach()
                    masks, valid_xyzb_sampled = self.assign_blocks_to_samples(xyz_sampled, ray_valid, is_train=is_train)
                    xyzb_sampled.masked_scatter_(ray_valid.unsqueeze(-1).expand(-1, -1, 4), valid_xyzb_sampled)
                    xyz_sampled = xyzb_sampled

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

            # compute sigma
            if (is_train and self.use_plane_split) or (not is_train and self.args.ckpt_type == "sub"):
                self.sigma_batch = masks[self.args.rank].sum()
                sigma_tensor_list = [
                    torch.zeros(mask.sum(), device=self.device)
                    if mask.sum() > 0 or not self.training
                    else torch.zeros((1), device=self.device)
                    for mask in masks
                ]
                if ray_valid.any() and masks[self.args.rank].any():
                    sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid][masks[self.args.rank]])
                    validsigma = self.feature2density(sigma_feature)
                else:
                    if self.training:
                        tmp_input = xyz_sampled[ray_valid][:1]
                        sigma_feature = self.compute_densityfeature(tmp_input)
                        validsigma = self.feature2density(sigma_feature)
                        sigma_tensor_list[self.args.rank] = torch.zeros((1), device=self.device)
                    else:
                        validsigma = sigma_tensor_list[self.args.rank]
                        # sigma[ray_valid] = validsigma

                with torch.no_grad():
                    dist.all_gather(sigma_tensor_list, validsigma)
                sigma_tensor_list[self.args.rank] = validsigma

                sigma_valid = torch.zeros_like(sigma[ray_valid], device=self.device)
                for mask, validsigma in zip(masks, sigma_tensor_list):
                    sigma_valid.masked_scatter_(mask, validsigma)
                sigma.masked_scatter_(ray_valid, sigma_valid)
            else:
                if ray_valid.any():
                    sigma_feature = self.compute_densityfeature(
                        xyz_sampled[ray_valid], use_xyzb=(not is_train and self.args.ckpt_type == "full")
                    )
                    validsigma = self.feature2density(sigma_feature)
                    sigma[ray_valid] = validsigma

            _, weight, _ = raw2alpha(sigma, dists * self.distance_scale)

            app_mask = weight > self.rayMarch_weight_thres
            # compute app
            if (is_train and self.use_plane_split) or (not is_train and self.args.ckpt_type == "sub"):
                valid_app_mask = (ray_valid & app_mask)[ray_valid]
                rgb_tensor_list = [
                    torch.zeros(((mask & valid_app_mask).sum(), 3), device=self.device)
                    if (mask & valid_app_mask).sum() > 0 or not self.training
                    else torch.zeros((1, 3), device=self.device)
                    for mask in masks
                ]
                self.app_batch = (valid_app_mask & masks[self.args.rank]).sum()

                if app_mask.any() and (valid_app_mask & masks[self.args.rank]).any():
                    app_features = self.compute_appfeature(
                        xyz_sampled[ray_valid][valid_app_mask & masks[self.args.rank]]
                    )
                    valid_rgbs = self.renderModule(
                        viewdirs[ray_valid][valid_app_mask & masks[self.args.rank]],
                        app_features,
                    )
                else:
                    if self.training:  # artifical sample is only needed in training mode
                        tmp_input = xyz_sampled[ray_valid][:1]
                        app_features = self.compute_appfeature(tmp_input)
                        valid_rgbs = self.renderModule(
                            viewdirs[ray_valid][:1],
                            app_features,
                        )
                        rgb_tensor_list[self.args.rank] = torch.zeros((1, 3), device=self.device)
                    else:
                        valid_rgbs = rgb_tensor_list[self.args.rank]

                with torch.no_grad():
                    dist.all_gather(rgb_tensor_list, valid_rgbs)
                rgb_tensor_list[self.args.rank] = valid_rgbs

                rgb_valid = torch.zeros_like(rgb[ray_valid], device=self.device)
                for mask, validrgb in zip(masks, rgb_tensor_list):
                    rgb_valid.masked_scatter_((valid_app_mask & mask).unsqueeze(-1).expand(-1, 3), validrgb)
                rgb.masked_scatter_(ray_valid.unsqueeze(-1).expand(-1, -1, 3), rgb_valid)
            else:
                if app_mask.any():
                    app_features = self.compute_appfeature(
                        xyz_sampled[app_mask], use_xyzb=(not is_train and self.args.ckpt_type == "full")
                    )
                    valid_rgbs = self.renderModule(viewdirs[app_mask], app_features)  # pylint: disable=E1102
                    rgb[app_mask] = valid_rgbs

            acc_map = torch.sum(weight, -1)
            rgb_map = torch.sum(weight[..., None] * rgb, -2)

            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1.0 - acc_map[..., None])
            rgb_map = rgb_map.clamp(0, 1)

            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, -1)

            if not is_train:
                outputs = {"rgb_map": rgb_map, "depth_map": depth_map}
            else:
                w = weight * app_mask
                m = torch.cat(
                    (
                        (z_vals[:, 1:] + z_vals[:, :-1]) * 0.5,
                        ((z_vals[:, 1:] + z_vals[:, :-1]) * 0.5)[:, -1:],
                    ),
                    dim=-1,
                )
                dist_loss = 0.01 * eff_distloss(w, m, dists).unsqueeze(0)
                outputs.update(
                    {
                        "rgb_map1": rgb_map,
                        "depth_map1": depth_map,
                        "distort_loss1": dist_loss,
                    }
                )

                # another round of weighted sample
                z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weight[..., 1:-1], self.nerf_n_importance)
                z_samples = z_samples.detach()

                # if (optional) second round of sample
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

                xyz_sampled = (
                    rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
                )  # [N_rays, N_samples, 3]

                if (is_train and self.use_plane_split) or (not is_train and self.args.ckpt_type == "sub"):
                    orig_xyz_sampled = xyz_sampled.clone().detach()
                    xyz_sampled = self.normalize_coord(xyz_sampled)
                    xyz_sampled = xyz_sampled.view(-1, 3)  # [N_rays*N_samples, 3]
                    # split samples
                    masks, xyz_sampled = self.split_samples(xyz_sampled)
                    self.sigma_batch = self.app_batch = self.nerf_batch = masks[self.args.rank].sum()
                    if masks[self.args.rank].sum():
                        sigma_feature, den_feat = self.compute_densityfeature(
                            xyz_sampled[masks[self.args.rank]], return_feat=True
                        )
                        app_features, app_feat = self.compute_appfeature(
                            xyz_sampled[masks[self.args.rank]], return_feat=True
                        )
                    else:
                        if self.training:  # artifical sample is only needed in training time
                            dummy_input = xyz_sampled[:1]
                            sigma_feature, den_feat = self.compute_densityfeature(dummy_input, return_feat=True)
                            app_features, app_feat = self.compute_appfeature(dummy_input, return_feat=True)
                        else:
                            den_feat = torch.zeros(
                                (masks[self.args.rank].sum(), sum(self.density_n_comp)),
                                device=self.device,
                            )
                            app_feat = torch.zeros(
                                (mask[self.args.rank].sum(), sum(self.app_n_comp)),
                                device=self.device,
                            )
                else:
                    sigma_feature, den_feat = self.compute_densityfeature(
                        self.normalize_coord(xyz_sampled).view(-1, 3),
                        return_feat=True,
                        use_xyzb=(not is_train and self.args.ckpt_type == "full"),
                    )
                    app_features, app_feat = self.compute_appfeature(
                        self.normalize_coord(xyz_sampled).view(-1, 3),
                        return_feat=True,
                        use_xyzb=(not is_train and self.args.ckpt_type == "full"),
                    )
                dists = torch.cat(
                    (z_vals[:, 1:] - z_vals[:, :-1], 1e3 * torch.ones_like(z_vals[:, :1])),
                    dim=-1,
                )
                if (is_train and self.use_plane_split) or (not is_train and self.args.ckpt_type == "sub"):
                    viewdirs = rays_d.view(-1, 1, 3).expand(orig_xyz_sampled.shape)
                    nray, npts = orig_xyz_sampled.shape[:2]
                    nerf_outputs_list = [
                        torch.zeros((mask.sum(), 4), device=self.device)
                        if mask.sum() or not self.training
                        else torch.zeros((1, 4), device=self.device)
                        for mask in masks
                    ]

                    if masks[self.args.rank].sum():
                        nerf_outputs = self.nerf(
                            orig_xyz_sampled.view(-1, 3)[masks[self.args.rank]],
                            viewdirs.reshape(-1, 3)[masks[self.args.rank]],
                            den_feat,
                            app_feat,
                            dists,
                        )
                    else:
                        if self.training:
                            nerf_outputs = self.nerf(
                                orig_xyz_sampled.view(-1, 3)[:1], viewdirs.reshape(-1, 3)[:1], den_feat, app_feat, dists
                            )
                            nerf_outputs_list[self.args.rank] = torch.zeros((1, 4), device=self.device)
                        else:
                            nerf_outputs = nerf_outputs_list[self.args.rank]

                    with torch.no_grad():
                        dist.all_gather(nerf_outputs_list, nerf_outputs)
                    nerf_outputs_list[self.args.rank] = nerf_outputs

                    all_outputs = torch.zeros((nray * npts, 4), device=self.device)
                    for mask, nerf_outputs in zip(masks, nerf_outputs_list):
                        all_outputs.masked_scatter_(mask.unsqueeze(-1).expand(-1, 4), nerf_outputs)
                    extras = raw2outputs(all_outputs.view(nray, npts, -1), dists)
                else:
                    viewdirs = rays_d.view(-1, 1, 3).expand(xyz_sampled.shape)
                    extras = self.nerf(xyz_sampled, viewdirs, den_feat, app_feat, dists)  # pylint: disable=E1102

                depth_map_nerf = torch.sum(extras["weights"] * z_vals, -1)

                if self.residnerf:
                    outputs.update(
                        {
                            "rgb_map_nerf": (extras["rgb_map"] + rgb_map).clamp(min=0.0, max=1.0),
                            "depth_map_nerf": depth_map_nerf,
                        }
                    )
                else:
                    outputs.update(
                        {
                            "rgb_map_nerf": extras["rgb_map"],
                            "depth_map_nerf": depth_map_nerf,
                        }
                    )

        extra_loss = {}
        # only compute extra_loss when training
        if is_train:
            if self.args.TV_weight_density > 0:
                extra_loss["TVloss_density"] = self.TV_loss_density(self.tvreg)
            if self.args.TV_weight_app > 0:
                extra_loss["TVloss_app"] = self.TV_loss_app(self.tvreg)
            if self.args.Ortho_weight > 0:
                extra_loss["vector_comp_diffs"] = self.vector_comp_diffs()
            if self.args.L1_weight_inital > 0:
                extra_loss["density_L1"] = self.density_L1()
        return outputs, extra_loss

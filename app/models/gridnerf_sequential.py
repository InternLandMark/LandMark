# pylint: disable=E1111
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from tools.dataloader.ray_utils import sample_pdf
from tools.utils import TVLoss, raw2alpha, st
from torch_efficient_distloss import eff_distloss

from .alpha_mask import AlphaGridMask
from .mlp_render_fea import MLPRender_Fea
from .nerf_branch import NeRF


class GridBaseSequential(torch.nn.Module):
    """
    Base class for GridNeRF

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
        use_plane_split=True,  # pylint: disable=W0613
        args=None,
        group=None,
        is_train=True,  # pylint: disable=W0613
    ):
        super().__init__()

        # new features
        if args.distributed:
            self.rank = args.rank
            self.world_size = args.world_size
            self.group = group

        self.train_eval = False
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

    def init_nerf(self, args):
        """
        create nerf branch
        """
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

    def compute_densityfeature(self, xyz_sampled, return_feat=False):
        pass

    def compute_appfeature(self, xyz_sampled, return_feat=False):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

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

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

            if ray_valid.any():
                xyz_sampled = self.normalize_coord(xyz_sampled)
                sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

                validsigma = self.feature2density(sigma_feature)
                sigma[ray_valid] = validsigma

            _, weight, _ = raw2alpha(sigma, dists * self.distance_scale)

            app_mask = weight > self.rayMarch_weight_thres

            if app_mask.any():
                app_features = self.compute_appfeature(xyz_sampled[app_mask])
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

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

            if ray_valid.any():
                xyz_sampled = self.normalize_coord(xyz_sampled)
                sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
                validsigma = self.feature2density(sigma_feature)
                sigma[ray_valid] = validsigma

            _, weight, _ = raw2alpha(sigma, dists * self.distance_scale)
            app_mask = weight > self.rayMarch_weight_thres
            if app_mask.any():
                app_features = self.compute_appfeature(xyz_sampled[app_mask])
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
                sigma_feature, den_feat = self.compute_densityfeature(
                    self.normalize_coord(xyz_sampled).view(-1, 3), return_feat=True
                )
                app_features, app_feat = self.compute_appfeature(
                    self.normalize_coord(xyz_sampled).view(-1, 3), return_feat=True
                )
                dists = torch.cat(
                    (z_vals[:, 1:] - z_vals[:, :-1], 1e3 * torch.ones_like(z_vals[:, :1])),
                    dim=-1,
                )
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


class GridNeRF(GridBaseSequential):
    """
    class for sequential version of GridNeRF
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
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp) * len(self.resMode), self.app_dim, bias=False).to(device)
        if self.nonlinear_density:
            self.basis_den = torch.nn.Linear(sum(self.density_n_comp) * len(self.resMode), 1, bias=False).to(device)

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
        if self.args.ckpt is None:  # set gridsize de novo
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                for j in self.resMode:
                    plane_coef.append(
                        torch.nn.Parameter(
                            scale
                            * torch.randn(
                                (
                                    1,
                                    n_component[i],
                                    gridSize[mat_id_1] * j,
                                    gridSize[mat_id_0] * j,
                                )
                            )
                        )
                    )  #
                    line_coef.append(
                        torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id] * j, 1)))
                    )
        else:
            # set gridsize according to ckpt, since the scale relationship between different planes is broken
            # when using merged ckpts trained by plane parallel
            ckpt = torch.load(self.args.ckpt, map_location=self.args.device)
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                for j in range(len(self.resMode)):
                    planeSize = ckpt["state_dict"][f"density_plane.{j}"].shape[
                        -2:
                    ]  # just need to use density plane to get both density/app plane size
                    plane_coef.append(
                        torch.nn.Parameter(scale * torch.randn((1, n_component[i], planeSize[0], planeSize[1])))
                    )
                    line_coef.append(
                        torch.nn.Parameter(
                            scale
                            * torch.randn(
                                (
                                    1,
                                    n_component[i],
                                    gridSize[vec_id] * self.resMode[j],
                                    1,
                                )
                            )
                        )
                    )
            del ckpt
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

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

    # obtain feature grids
    def compute_densityfeature(self, xyz_sampled, return_feat=False):
        """
        Computes the density feature for a given set of sampled points.

        Args:
            xyz_sampled (torch.Tensor): The sampled points.
            return_feat (bool): Whether to return the feature.

        Returns:
            torch.Tensor: The density feature.
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
            plane_coef = F.grid_sample(
                self.density_plane[idx_plane],
                coordinate_plane[[idx_dim]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            line_coef = F.grid_sample(
                self.density_line[idx_plane],
                coordinate_line[[idx_dim]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            if self.nonlinear_density or return_feat:
                plane_coef_point.append(plane_coef)
                line_coef_point.append(line_coef)
            sigma_feature = sigma_feature + torch.sum(plane_coef * line_coef, dim=0)
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        if self.nonlinear_density:
            sigma_feature = F.relu(self.basis_den((plane_coef_point * line_coef_point).T))[  # pylint: disable=E1102
                ..., 0
            ]

        if return_feat:
            return sigma_feature, (plane_coef_point * line_coef_point).T

        return sigma_feature

    def compute_appfeature(self, xyz_sampled, return_feat=False):
        """
        Computes the apperance feature for a given set of sampled points.

        Args:
            xyz_sampled (torch.Tensor): The sampled points.
            return_feat (bool): Whether to return the feature.

        Returns:
            torch.Tensor: The appearance feature.
        """
        N = self.ndims
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
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
                F.grid_sample(
                    self.app_line[idx_plane],
                    coordinate_line[[idx_dim]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        if return_feat:
            return (
                self.basis_mat((plane_coef_point * line_coef_point).T),  # pylint: disable=E1102
                (plane_coef_point * line_coef_point).T,
            )
        return self.basis_mat((plane_coef_point * line_coef_point).T)  # pylint: disable=E1102

    # upsample/update grids
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        """
        Upsamples the plane and line coefficients to a target resolution.

        Args:
            plane_coef (list): The plane coefficients.
            line_coef (list): The line coefficients.
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

        Args:
            vector_comps (list): A list of vectors.

        Returns:
            torch.Tensor: The mean absolute difference between non-diagonal elements of the dot product of \
                the input vectors.
        """
        total = 0
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
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        """
        Computes the L1 loss for density plane and density line

        Returns:
            torch.Tensor: The loss value
        """
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))
            )
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
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2
        return total

    def TV_loss_app(self, reg):
        """
        Computes the total variation loss for the appearance plane coefficients.

        Args:
            reg (function): The regularization function.

        Returns:
            torch.Tensor: The total variation loss for the appearance plane coefficients.
        """
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2
        return total

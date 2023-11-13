# pylint: disable=E1111,E1102
import torch

from app.tools.utils import raw2alpha
from dist_render.comm.global_memory_buffer import GlobalMemoryBuffer

from .gridnerf_sequential import GridBaseSequential
from .nerf_branch import NeRFParallel


class Buffer:
    """
    Desc:
        Buffer contains metadata of local space and planes/lines of local space.
    """

    def __init__(
        self,
        center: torch.Tensor = None,
        center_block_idx: int = None,
        density_plane: torch.nn.ParameterList = None,
        app_plane: torch.nn.ParameterList = None,
        density_line: torch.nn.ParameterList = None,
        app_line: torch.nn.ParameterList = None,
        neighbours: list = None,
    ):
        """
        Desc:
            Initialize the buffer.

        Args:
            center (torch.Tensor): The coordinates of local space center in global space.
            center_block_idx (int): The central block index of local plane in global plane.
            neighbours (list): List of total block indexes of local plane in global plane.
            density_plane (torch.nn.ParameterList): The density feature local plane.
            app_plane (torch.nn.ParameterList): The appearance feature local plane.
            density_line (torch.nn.ParameterList): The line corresponding to the density plane.
            app_line (torch.nn.ParameterList): The line corresponding to the appearance plane.
        """
        # metadata
        self.center = center
        self.center_block_idx = center_block_idx
        self.neighbours = neighbours

        self.density_plane = density_plane
        self.app_plane = app_plane
        self.density_line = density_line
        self.app_line = app_line


class DistRenderGridNeRFElasticBase(GridBaseSequential):
    """
    Class for Dynamic Fetching version of TensorBase.
    The difference between Dynamic Fetching with other is that
    it doesn't load all the model parameter on GPU. Instead, It
    only load just enough parameter so that we can render the
    correct picture, and use heuristic method to preload
    other parameters.
    """

    def __init__(self, aabb, gridSize, device, neighbour_size, **kargs):
        self.neighbour_size = neighbour_size
        assert self.neighbour_size > 0
        self.neighbour_width = int(self.neighbour_size**0.5 + 0.01)
        assert self.neighbour_size == self.neighbour_width**2
        self.current_buffer_idx = 0  # 0/1
        self.buffers = [Buffer(), Buffer()]
        self.bound = torch.cuda.FloatTensor([0.0, 2.0, 0.0, 2.0])
        self.mem_buffer = GlobalMemoryBuffer()

        super().__init__(aabb, gridSize, device, **kargs)

    def init_nerf(self, args):
        """
        create nerf branch
        """
        self.nerf = NeRFParallel(
            args,
            sum(self.density_n_comp) * len(self.resMode),
            sum(self.app_n_comp) * len(self.resMode),
        ).to(self.device)
        self.residnerf = args.residnerf
        self.n_importance = args.n_importance
        self.nerf_n_importance = args.nerf_n_importance
        self.run_nerf = True

    def get_left_top_block_pos(self):
        """
        get the top left block's position
        """
        radius = self.neighbour_width // 2
        center_block_idx = self.buffers[self.current_buffer_idx].center_block_idx
        corner_block_idx_x = center_block_idx // self.args.plane_division[1] - radius
        corner_block_idx_y = center_block_idx % self.args.plane_division[1] - radius
        return (corner_block_idx_x, corner_block_idx_y)

    def assign_blocks_to_samples(self, xyz_sampled, ray_valid=None):
        """
        Desc:
            Assign samples to corresponding blocks according to x-y coordinates.
            Append the blocks index after xyz coords, compositing a 4D coords(xyzb, b:block index)
            Used when rendering a model trained by block parallel.

        Args:
            xyz_sampled: Samples after normalized to [-1,1] by normalize_coord.
            ray_valid: The mask to filter the valid sample points.
        """
        if ray_valid is not None:
            valid_xyz_sampled = xyz_sampled[ray_valid].clone().detach()
        else:
            valid_xyz_sampled = xyz_sampled.clone().detach()
        # generate masks
        plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]

        valid_xyzb_sampled = self.mem_buffer.get_tensor(
            (*valid_xyz_sampled.shape[:-1], 4),
            dtype=valid_xyz_sampled.dtype,
            name="assign_blocks_to_samples.valid_xyzb_sampled.1",
        )
        valid_xyzb_sampled[..., :3] = valid_xyz_sampled

        valid_xyzb_sampled[..., :2] += 1
        valid_xyzb_sampled[..., 0] /= plane_width
        valid_xyzb_sampled[..., 1] /= plane_height
        valid_xyzb_sampled[..., 3] = torch.floor(valid_xyzb_sampled[..., 0]) * self.args.plane_division[
            1
        ] + torch.floor(valid_xyzb_sampled[..., 1])
        valid_xyzb_sampled[..., :3] = valid_xyz_sampled

        # normalized xyb coords
        coord_min = self.mem_buffer.get_tensor(
            (*valid_xyzb_sampled.shape[:-1], 2),
            dtype=valid_xyz_sampled.dtype,
            name="assign_blocks_to_samples.coord_min.1",
        )
        torch.floor_divide(valid_xyzb_sampled[..., 3], self.args.plane_division[1], out=coord_min[..., 0])
        torch.remainder(valid_xyzb_sampled[..., 3], self.args.plane_division[1], out=coord_min[..., 1])
        coord_min[..., 0].mul_(plane_width)
        coord_min[..., 1].mul_(plane_height)
        coord_min.sub_(1)

        # calculate current center block's x index and y index.
        corner_block_idx_x, corner_block_idx_y = self.get_left_top_block_pos()

        # assin local blocks' index to samples
        valid_xyzb_sampled[..., :2] += 1
        valid_xyzb_sampled[..., 0] /= plane_width
        valid_xyzb_sampled[..., 0] -= corner_block_idx_x
        valid_xyzb_sampled[..., 1] /= plane_height
        valid_xyzb_sampled[..., 1] -= corner_block_idx_y
        valid_xyzb_sampled[..., 3] = torch.floor(valid_xyzb_sampled[..., 0]) * self.neighbour_width + torch.floor(
            valid_xyzb_sampled[..., 1]
        )

        valid_xyzb_sampled[..., :3] = valid_xyz_sampled

        valid_xyzb_sampled[..., :2] = (valid_xyzb_sampled[..., :2] - coord_min) * 2 / torch.tensor(
            [plane_width, plane_height], dtype=torch.float32, device=torch.cuda.current_device()
        ) - 1

        masks = []
        if self.args.ckpt_type == "sub":  # no need when using full ckpt
            for block_idx in range(self.args.plane_division[0] * self.args.plane_division[1]):
                masks.append(valid_xyzb_sampled[..., 3] == block_idx)

        valid_xyzb_sampled[..., 3] = -1 + 2 * valid_xyzb_sampled[..., 3] / (self.neighbour_size - 1)

        return masks, valid_xyzb_sampled

    def compute_app_latent(self, xyzb_sampled, app_mask):
        if self.args.encode_app:
            fake_xyzb_sampled_idxs = torch.zeros(xyzb_sampled.shape[:-1], dtype=torch.long, device=self.device)
            app_latent = self.embedding_app(fake_xyzb_sampled_idxs[app_mask])
            del fake_xyzb_sampled_idxs
        else:
            app_latent = None
        return app_latent

    def app_render_full(self, xyzb_sampled, app_mask, viewdirs, rgb):
        app_features = self.compute_appfeature(xyzb_sampled[app_mask], use_xyzb=True)  # pylint: disable=E1123
        app_latent = self.compute_app_latent(xyzb_sampled, app_mask)
        valid_rgbs = self.renderModule(
            viewdirs[app_mask],
            app_features,
            app_latent,
        )
        rgb[app_mask] = valid_rgbs
        return rgb, None, None

    @torch.no_grad()
    def forward(
        self,
        rays_chunk,
        white_bg=True,
        is_train=False,
        N_samples=-1,
    ):
        rays_o = rays_chunk[:, :3]
        viewdirs = rays_chunk[:, 3:6]

        xyz_sampled, _, z_vals, ray_valid = self.sample_ray_within_hull(
            rays_o,
            viewdirs,
            is_train=is_train,
            N_samples=self.n_importance,
        )
        # ensure that the sampled points are within the neighbour blocks && the entire map
        xyz_norm = self.mem_buffer.get_tensor(xyz_sampled.shape, xyz_sampled.dtype, "forward.xyz_norm.1")
        xyz_norm.copy_(xyz_sampled)
        xyz_norm.sub_(self.aabb[0]).mul_(self.invaabbSize).sub_(1)

        plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]
        corner_block_idx_x, corner_block_idx_y = self.get_left_top_block_pos()

        boundingbox = self.mem_buffer.get_tensor(xyz_norm.shape[:-1], torch.bool, "forward.boundingbox.1")
        torch.le(
            xyz_norm[..., 0], min(2.0, (corner_block_idx_x + self.neighbour_width) * plane_width) - 1, out=boundingbox
        )
        ray_valid.logical_and_(boundingbox)
        torch.ge(xyz_norm[..., 0], max(0.0, corner_block_idx_x * plane_width) - 1, out=boundingbox)
        ray_valid.logical_and_(boundingbox)
        torch.le(
            xyz_norm[..., 1], min(2.0, (corner_block_idx_y + self.neighbour_width) * plane_height) - 1, out=boundingbox
        )
        ray_valid.logical_and_(boundingbox)
        torch.ge(xyz_norm[..., 1], max(0.0, corner_block_idx_y * plane_height) - 1, out=boundingbox)
        ray_valid.logical_and_(boundingbox)

        dists = self.mem_buffer.get_tensor(z_vals.shape, z_vals.dtype, "forward.dists.1")
        torch.sub(z_vals[:, 1:], z_vals[:, :-1], out=dists[:, :-1])
        dists[:, -1] = 0
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        sigma = self.mem_buffer.get_tensor(xyz_sampled.shape[:-1], xyz_sampled.dtype, "forward.sigma.1")
        sigma.zero_()
        rgb = self.mem_buffer.get_tensor((*xyz_sampled.shape[:2], 3), xyz_sampled.dtype, "forward.rgb.1")
        rgb.zero_()

        xyz_sampled = xyz_norm

        assert self.args.ckpt_type == "full"
        # assign samples
        xyzb_sampled = self.mem_buffer.get_tensor(
            (*xyz_sampled.shape[:-1], 4), xyz_sampled.dtype, "forward.xyzb_sampled.1"
        )
        xyzb_sampled[..., :3] = xyz_sampled
        xyzb_sampled[..., 3] = 0

        _, valid_xyzb_sampled = self.assign_blocks_to_samples(xyz_sampled, ray_valid)
        xyzb_sampled.masked_scatter_(ray_valid.unsqueeze(-1).expand(-1, -1, 4), valid_xyzb_sampled)

        if ray_valid.any():

            sigma_feature = self.compute_densityfeature(xyzb_sampled[ray_valid], use_xyzb=True)  # pylint: disable=E1123
            validsigma = self.feature2density(sigma_feature)

            sigma[ray_valid] = validsigma
        else:
            pass
        _, weight, _ = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = self.mem_buffer.get_tensor(weight.shape, torch.bool, "forward.app_mask.1")
        torch.gt(weight, self.rayMarch_weight_thres, out=app_mask)

        if self.args.ckpt_type == "full":
            if app_mask.any():
                rgb, _, _ = self.app_render_full(xyzb_sampled, app_mask, viewdirs, rgb)

        elif self.args.ckpt_type == "sub":
            raise NotImplementedError
        elif self.args.ckpt_type == "part":
            raise NotImplementedError
        else:
            assert False, f"Invalid argument, args.ckpt_type must be full/part/sub, but get {self.args.ckpt_type}"

        acc_map = self.mem_buffer.get_tensor(weight.shape[:-1], weight.dtype, "forward.acc_map.1")
        torch.sum(weight, -1, out=acc_map)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)

        depth_map = torch.sum(weight * z_vals, -1)

        outputs = {"rgb_map": rgb_map, "depth_map": depth_map}

        extra_loss = {}
        return outputs, extra_loss

# pylint: disable=E1111,E1102
import AssignBlocksToSamples_dal
import compute_beta_dal
import compute_weight_dal
import grid_sampler_ndhwc_dal
import pipeline_expand_index_encoding_mlp_dal
import SamplerayGridsample_dal
import torch

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


class DistRenderGridNeRFElasticCudaBase(GridBaseSequential):
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

    def tensorf_part2_cuda_bz_columnmajor(
        self, tensorf, rays_chunk, weight_top, app_mask, appfeature_cuda_half, plane_mul_line_T
    ):
        N_rays = rays_chunk.shape[0]

        valid_samples_num = app_mask.size(1)

        if tensorf.args.encode_app:
            app_latent_shape = 1 if tensorf.pipeline else appfeature_cuda_half.shape[1]  # xyz_sampled.shape[:-1]
            fake_xyzb_sampled_idxs = self.fake_xyzb_sampled_idxs.view(-1)[:app_latent_shape].view(app_latent_shape)
            app_latent = tensorf.embedding_app(fake_xyzb_sampled_idxs)  # fake_xyzb_sampled_idxs[app_mask]
        else:
            app_latent = None

        with torch.no_grad():
            index = tensorf.index_ori[:N_rays].expand(-1, valid_samples_num)  # index.expand(-1, valid_samples_num)
            index = index[app_mask]

            # to improve, tip: to avoid the conversion
            features = (
                appfeature_cuda_half.t()
            )  # appfeature_cuda.cuda().half().t().contiguous().t() Notice: half + column major

            if tensorf.pipeline:
                m = features.shape[0]
                if tensorf.pipeline_tmp_m > m:
                    tmp_m = m - m % 8
                else:
                    tmp_m = tensorf.pipeline_tmp_m

                mlp_layer3_out_top = tensorf.mlp_layer3_out.view(-1)[: m * tensorf.n3].view(m, tensorf.n3)
                pipeline_expand_index_encoding_mlp_dal.app_feature_expand_encoding_gemm_fp16_user_defined(
                    plane_mul_line_T=plane_mul_line_T,
                    basis_mat_weight=tensorf.basis_mat.weight.half(),
                    fea_tmp=tensorf.mlp_layerout_tmp,
                    gemm_app_n=tensorf.gemm_app_n,
                    gemm_app_k=tensorf.gemm_app_k,
                    gemm_app_activation="None",
                    app_latent=app_latent,
                    app_offset=0,
                    viewdirs=rays_chunk,
                    n_freque_fea=tensorf.args.fea_pe,
                    n_freque_view=tensorf.args.view_pe,
                    fea_encoding_offset=tensorf.feature_encoding_offset + app_latent.size(1),
                    view_encoding_offset=tensorf.view_encoding_offset + app_latent.size(1),
                    fea_data_offset=tensorf.fea_offset + app_latent.size(1),
                    view_data_offset=tensorf.view_offset + app_latent.size(1),
                    padding_offset=150 + app_latent.size(1),
                    padding_width=2,
                    view_in_offset=3,
                    view_valid_column=3,
                    index_view=index,
                    tmp_m=tmp_m,
                    tmp_in_rowmajor=False,
                    tmp_layer1out_rowmajor=False,
                    tmp_layer2out_rowmajor=False,
                    output_rowmajor=True,
                    m=m,
                    n1=tensorf.n1,
                    k1=tensorf.k1,
                    activation1="Relu",
                    n2=tensorf.n2,
                    k2=tensorf.k2,
                    activation2="Relu",
                    n3=tensorf.n3,
                    k3=tensorf.k3,
                    activation3="Sigmoid",
                    weight1=tensorf.layer1_weight_paded,
                    weight2=tensorf.renderModule.mlp[2].weight.half(),
                    weight3=tensorf.layer3_weight_paded,
                    tmp_mlp_in=tensorf.mlp_in_tmp,
                    tmp_layer1out=tensorf.mlp_layerout_tmp,
                    tmp_layer2out=tensorf.mlp_layerout_tmp,
                    output=mlp_layer3_out_top,
                    multiStream=False,
                )

            rgb = tensorf.rgb.view(-1)[: N_rays * valid_samples_num * 3].view(N_rays, valid_samples_num, 3)
            rgb[app_mask] = mlp_layer3_out_top[:, :3]

            rgb_map_cuda = torch.sum(weight_top[..., None] * rgb, -2)

            outputs = {
                "mlp_layer3_out_top": mlp_layer3_out_top,  # differ across chunk
                "rgb_map_cuda": rgb_map_cuda,  # differ across chunk
            }

        return outputs

    @torch.no_grad()
    def forward(
        self,
        rays_chunk,
        white_bg=True,
        is_train=False,
        N_samples=-1,
    ):  # pylint: disable=W0613
        cuda_N_rays = rays_chunk.shape[0]

        SamplerayGridsample_dal.cuda(
            rays_chunk,
            torch.tensor(self.aabb).cuda(),
            torch.tensor(self.aabb).cuda(),
            torch.tensor(self.near_far).cuda(),
            False,
            self.n_importance,
            self.valid_samples_num,
            self.xyz_sampled_cuda,
            self.z_vals_cuda,
            self.maskbbox_cuda,
            self.tvals_min_cuda,
            self.tvals_min_index_cuda,
            self.density_plane_line_sum,
            self.sigma_feature_bool_cuda,
            0,
            False,
            False,
            512,
            self.n_importance,
            1024,
        )

        valid_samples_num = self.valid_samples_num[0].item()
        valid_samples_num = max(valid_samples_num, 0)

        xyz_sample_valid = self.xyz_sampled_cuda.view(-1)[: cuda_N_rays * valid_samples_num * 3].view(
            cuda_N_rays, valid_samples_num, 3
        )
        z_vals_top = self.z_vals_cuda.view(-1)[: cuda_N_rays * valid_samples_num].view(cuda_N_rays, valid_samples_num)
        ray_valid = self.sigma_feature_bool_cuda.view(-1)[: cuda_N_rays * valid_samples_num].view(
            cuda_N_rays, valid_samples_num
        )

        sigma_feature_valid_cuda = self.sigma_feature_valid_cuda[: xyz_sample_valid[ray_valid].shape[0]]
        b_sampled_cuda_cut = self.b_sampled_cuda.view(-1)[: cuda_N_rays * valid_samples_num].view(
            cuda_N_rays, valid_samples_num
        )
        if ray_valid.any():
            int_mode = True
            debug_mode = False
            valid_samples_num_max = 128
            num_block = 512

            corner_block_idx_x, corner_block_idx_y = self.get_left_top_block_pos()
            AssignBlocksToSamples_dal.cuda_norm_relative(
                xyz_sample_valid,
                ray_valid,
                b_sampled_cuda_cut,
                ray_valid,
                self.args.plane_division[0],
                self.args.plane_division[1],
                corner_block_idx_x,
                corner_block_idx_y,
                self.neighbour_width,
                int_mode,
                debug_mode,
                num_block,
                valid_samples_num_max,
            )
            JudgeOverflow = True

            grid_sampler_ndhwc_dal.gridsample_sum_xyz_b3_noMalloc_nbhwc(
                xyz_sample_valid[ray_valid],
                b_sampled_cuda_cut[ray_valid],
                self.density_plane,
                self.density_line,
                self.hw_in,
                self.plane_line_ptr,
                sigma_feature_valid_cuda,
                debug_mode,
                num_block,
                JudgeOverflow,
            )
        else:
            pass

        xyz_sampled_alpha_mask = xyz_sample_valid[ray_valid]

        sigma_feature = sigma_feature_valid_cuda.view(-1)[: xyz_sampled_alpha_mask.shape[0]]

        validsigma_cuda = self.feature2density(sigma_feature)
        sigma_feature_cuda_cut = self.sigma_feature_cuda.view(-1)[: cuda_N_rays * valid_samples_num].view(
            cuda_N_rays, valid_samples_num
        )
        sigma_feature_cuda_cut = torch.zeros_like(sigma_feature_cuda_cut)
        sigma_feature_cuda_cut[ray_valid] = validsigma_cuda

        z_vals_cuda_cut = z_vals_top.view(-1)[: cuda_N_rays * valid_samples_num].view(cuda_N_rays, valid_samples_num)

        sigma_cuda = torch.nn.functional.relu(sigma_feature_cuda_cut)
        sigma_cuda = sigma_cuda.view(cuda_N_rays, -1)
        compute_beta_dal.cuda(sigma_cuda, z_vals_cuda_cut, self.beta, self.distance_scale)

        beta_cut = self.beta.view(-1)[: cuda_N_rays * valid_samples_num].view(cuda_N_rays, valid_samples_num)
        compute_weight_dal.cuda(beta_cut, self.weight_cuda)

        self.z_vals_top = z_vals_cuda_cut
        self.weight_top = self.weight_cuda.view(-1)[: cuda_N_rays * valid_samples_num].view(
            cuda_N_rays, valid_samples_num
        )

        self.app_mask = self.weight_top > self.rayMarch_weight_thres

        xyz_sampled_app_mask = xyz_sample_valid[self.app_mask]
        b_sampled_app_mask = b_sampled_cuda_cut[self.app_mask]

        if self.app_mask.any():
            plane_mul_line_list = self.plane_mul_line_list.view(-1)[
                : xyz_sampled_app_mask.shape[0] * len(self.app_plane) * self.grid_channel
            ].view(xyz_sampled_app_mask.shape[0], len(self.app_plane), self.grid_channel)
            serial = 8
            grid_sampler_ndhwc_dal.gridsample_ew_xyb_bz_nbhwc2(
                xyz_sampled_app_mask, b_sampled_app_mask, self.app_plane, self.app_line, plane_mul_line_list, serial, 0
            )

            appfeature_cuda_half = self.appfeature_cuda_half.view(-1)[
                : self.basis_mat.weight.shape[0] * xyz_sampled_app_mask.shape[0]
            ].view(self.basis_mat.weight.shape[0], xyz_sampled_app_mask.shape[0])
            plane_mul_line_T_rm = plane_mul_line_list
            ret_cuda_part2 = self.tensorf_part2_cuda_bz_columnmajor(
                self, rays_chunk, self.weight_top, self.app_mask, appfeature_cuda_half, plane_mul_line_T_rm
            )
            rgb_map = ret_cuda_part2["rgb_map_cuda"]  # differ across chunk
            outputs = {"rgb_map": rgb_map}
        else:
            outputs = {"rgb_map": torch.zeros((rays_chunk.shape[0], 3), dtype=torch.float32, device=self.device)}

        return outputs

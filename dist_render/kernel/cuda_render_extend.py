import math
from abc import abstractmethod

import AssignBlocksToSamples
import compute_appfeature
import compute_beta
import compute_weight
import gemm_3xtf32_fast_accurate_GaColumnMajor as gemm_gac
import numpy as np
import pe_concate as pc
import SamplerayGridsample
import torch

# import pipeline_expand_index_encoding_mlp
# import pe_column_major_half2half as fe
# import expand_encoding_2half as expand_encoding #new

mlp_dtype_ = torch.half  # torch.float
if mlp_dtype_ == torch.float:
    import gemm_fp32 as gemm
else:
    import gemm_fp16 as gemm
print(gemm)


# Codes copy from fr cuda_render_extend
class AbstractMultiBlockParallelCuda:
    """
    Abstract class of cuda implementation.
    """

    def __init__(self) -> None:
        self._tensorf = None

    def allocate_tensors(self, tensorf, N_rays):
        """
        allocate cuda tensors in advance.

        Args:
            tensorf(Module): nerf module.
            N_rays(int): rays number.
        """
        self._tensorf = tensorf
        N_samples = self._tensorf.n_importance  # 128

        # malloc the maximum number of valid samples, i.e., 128
        Max_samples_num = self._tensorf.n_importance  # 64  # 128 #self._tensorf.valid_samples_num #

        # by LHD
        self._tensorf.tvals_min_index_cuda = torch.ones(1).to(self._tensorf.device)
        self._tensorf.tvals_min_cuda = torch.rand(math.ceil(N_rays / 1024)).to(self._tensorf.device)

        self._tensorf.near_far = torch.from_numpy(np.array(self._tensorf.near_far)).float().to(self._tensorf.device)

        self._tensorf.z_vals_cuda = torch.rand((N_rays, Max_samples_num)).to(self._tensorf.device)

        self._tensorf.assign_masks_cuda = torch.ones(1).to(self._tensorf.device) == 0
        # plane_x = self._tensorf.args.plane_division[0]  # 4
        # plane_y = self._tensorf.args.plane_division[1]  # 6
        # self._tensorf.assign_masks_cuda = (
        #     torch.randint(2, [plane_x, plane_y, N_rays * Max_samples_num]).to(self._tensorf.device)
        #     == 0
        # )
        self._tensorf.valid_b_sampled_cuda = (
            torch.zeros(  # torch.rand(N_rays, Max_samples_num).to(self._tensorf.device)
                [N_rays, N_samples], dtype=torch.int
            ).to(self._tensorf.device)
        )
        self._tensorf.xyz_sampled_cuda = torch.rand(N_rays, Max_samples_num, 3).to(self._tensorf.device)

        # not used
        self._tensorf.maskbbox_cuda = torch.randint(2, [N_rays, N_samples]).to(self._tensorf.device) == 0

        # for alpha
        self._tensorf.sigma_feature_bool_cuda = (
            torch.randint(  # randbooltrue(
                2,
                [
                    1,
                    1,
                    N_rays,
                    Max_samples_num,
                    1,
                ],
            ).to(self._tensorf.device)
            == 0
        )
        # for density
        self._tensorf.sigma_feature_cuda = torch.rand(
            1, 1, N_rays, Max_samples_num, 1  # density_plane_line_sum.shape[1],  # int(N_samples*(1-tvals_min_cuda[0]))
        ).to(self._tensorf.device)

        # by ZCQ
        self._tensorf.sigma_cuda = torch.zeros((N_rays, Max_samples_num)).to(
            self._tensorf.device
        )  # , device=rays_chunk.device)
        self._tensorf.beta = torch.empty((N_rays, Max_samples_num)).to(
            self._tensorf.device
        )  # , device=rays_chunk.device)
        self._tensorf.weight_cuda = torch.empty((N_rays, Max_samples_num)).to(
            self._tensorf.device
        )  # , device=rays_chunk.device)

        # by XHR
        # appfeature_cuda = [90w, 27]
        m = N_rays * 64  # 32
        self._tensorf.density_feature_cuda = torch.zeros((m * 4)).to(self._tensorf.device)
        grid_channel = self._tensorf.app_plane[0].shape[1]
        self._tensorf.plane_mul_line_list = torch.zeros(
            (len(self._tensorf.app_plane), (grid_channel * m)), device=self._tensorf.device
        )

        # by SZL

        # encoding and expand index place
        self._tensorf.fea_offset = 0
        self._tensorf.view_offset = 27  # 39 #
        self._tensorf.view_encoding_offset = self._tensorf.view_offset + 3  # 27 #
        self._tensorf.feature_encoding_offset = 42

        # shape: [128, 198]
        weight_half = self._tensorf.renderModule.mlp[0].weight.half()
        encappdim = self._tensorf.embedding_app.embedding_dim

        # permutation：
        # new mlp_in = [latent, features, viewdirs,
        #           positional_encoding(features, self.feape),
        #           positional_encoding(viewdirs, self.viewpe)]
        # Concatenate in 2nd dimension
        self._tensorf.layer1_weight_paded = torch.cat(
            [weight_half[:, 27 : 27 + encappdim], weight_half[:, :27], weight_half[:, 27 + encappdim :]], dim=1
        )
        # print(self._tensorf.layer1_weight_paded.shape)
        self._tensorf.layer1_weight_paded = torch.nn.functional.pad(self._tensorf.layer1_weight_paded, [0, 2, 0, 0])

        # layer3_weight_paded: [3, 128] ---> [8, 128]
        self._tensorf.layer3_weight_paded = torch.nn.functional.pad(
            self._tensorf.renderModule.mlp[4].weight.half(), [0, 0, 0, 5]
        )

        basis_mat_weight = self._tensorf.basis_mat.weight  # basis_mat_weight.shape[0] = 144
        self._tensorf.appfeature_cuda = torch.empty((m, basis_mat_weight.shape[0]), device=self._tensorf.device)

        # by SZL
        self._tensorf.pc_cuda = torch.empty((m, 152), device=self._tensorf.device).half()

        # k1 = 152; # if k1 = 150
        n1 = self._tensorf.layer3_weight_paded.size(1)
        # n2 = self._tensorf.layer3_weight_paded.size(1)
        # k2 = 128;
        # n2 = 128; # k3 = 128;
        self._tensorf.mlp_layer1_out = torch.empty((m, n1), dtype=mlp_dtype_).to(self._tensorf.device)
        # self._tensorf.mlp_layer2_out = torch.empty((m , n2), dtype = mlp_dtype_).to(self._tensorf.device)

        n3 = self._tensorf.layer3_weight_paded.size(0)  # if n3 = 3, RuntimeError: CUDA error: misaligned address
        self._tensorf.mlp_layer3_out = torch.empty((m, n3), dtype=mlp_dtype_).to(self._tensorf.device)

        # rgb: [chunk_cuda, 50, 3]
        self._tensorf.rgb = torch.zeros((N_rays, Max_samples_num, 3), dtype=mlp_dtype_).to(self._tensorf.device)

    @abstractmethod
    def forward(self):
        pass


class MultiBlockParallelCuda(AbstractMultiBlockParallelCuda):
    """
    tensorf_multiBlockParallel_cuda_v0606
    """

    def __init__(self) -> None:
        super().__init__()
        # self._tensorf.fullgrid = 0
        # 0 for v0606
        # 1 for v0607
        # self._tensorf.column_major = 0

    def tensorf_multiBlockParallel_cuda_part0(self, rays_chunk):
        """
        sample rays and assign blocks to samples with multi blocks .
        """
        aabb = self._tensorf.aabb
        near_far = self._tensorf.near_far  # torch.from_numpy(np.array(tensorf.near_far)).float().to(rays_chunk.device)
        N_samples = self._tensorf.n_importance  # 128
        N_rays = rays_chunk.shape[0]

        # for sampleray
        valid_samples_num_max = N_samples  # 128
        debug_mode = False
        caltvalmin_simple = False

        density_plane_line_sum = self._tensorf.alphaMask.alpha_volume

        tvals_min_index_cuda = self._tensorf.tvals_min_index_cuda
        tvals_min_cuda = self._tensorf.tvals_min_cuda
        xyz_sampled_cuda = self._tensorf.xyz_sampled_cuda
        z_vals_cuda = self._tensorf.z_vals_cuda
        maskbbox_cuda = self._tensorf.maskbbox_cuda.fill_(True)
        sigma_feature_bool_cuda = self._tensorf.sigma_feature_bool_cuda

        with torch.no_grad():
            # valid_samples_num = -1
            valid_samples_num = torch.from_numpy(np.array([-1]))
            # valid_samples_num = torch.from_numpy(np.array([256]))

            SamplerayGridsample.cuda(
                rays_chunk,
                aabb,
                self._tensorf.alphaMask.aabb,
                near_far,
                False,
                N_samples,
                valid_samples_num,
                xyz_sampled_cuda,
                z_vals_cuda,
                maskbbox_cuda,
                tvals_min_cuda,
                tvals_min_index_cuda,
                density_plane_line_sum,
                sigma_feature_bool_cuda,
                self._tensorf.alpha_mask_filter_thre,
                caltvalmin_simple,
                debug_mode,
                512,
                valid_samples_num_max,
                1024,
            )

            valid_samples_num = valid_samples_num[0].item()
            valid_samples_num = max(valid_samples_num, 0)

            # valid_samples_num = N_samples - int(tvals_min_index_cuda[0].cpu().item())

            # assign samples
            # xyz_sample_valid shape: [N_rays, valid_samples_num, 3]
            xyz_sample_valid = xyz_sampled_cuda.view(-1)[: N_rays * valid_samples_num * 3].view(
                N_rays, valid_samples_num, 3
            )
            # xyz_sample_valid_norm = xyz_sample_valid.clone()

            # valid_b_sampled shape: [N_rays, valid_samples_num, 1]
            valid_b_sampled = self._tensorf.valid_b_sampled_cuda.view(-1)[: N_rays * valid_samples_num * 1].view(
                N_rays, valid_samples_num
            )  # .int()

            ray_valid = sigma_feature_bool_cuda.view(-1)[: N_rays * valid_samples_num].view(N_rays, valid_samples_num)

            int_mode = True
            AssignBlocksToSamples.cuda_norm(
                xyz_sample_valid,
                ray_valid,
                valid_b_sampled,
                self._tensorf.assign_masks_cuda,
                self._tensorf.args.plane_division[0],
                self._tensorf.args.plane_division[1],
                int_mode,
                debug_mode,
            )

            # # valid_xyzb_sampled_cuda shape: [N_rays, valid_samples_num, 4]
            # xyzb_sampled = torch.cat((xyz_sample_valid, valid_b_sampled.view(N_rays, valid_samples_num, 1)),
            #                         2) #1)

            # by ZCQ
            # xyz_sampled_top = xyz_sampled_cuda.view(-1)[: N_rays * valid_samples_num * 3].view(
            #     N_rays, valid_samples_num, 3
            # )

            self._tensorf.z_vals_top = z_vals_cuda.view(-1)[: N_rays * valid_samples_num].view(
                N_rays, valid_samples_num
            )

            outputs = {
                "xyz_sampled_cuda": xyz_sample_valid,  # xyz_sampled_cuda,
                "z_vals_cuda": self._tensorf.z_vals_top,  # z_vals_cuda,
                "valid_b_sampled": valid_b_sampled,
                "ray_valid": ray_valid,
            }
        return outputs

    def compute_densityfeature_xyzb_cuda(
        self,
        xyz_sampled,
        b_sampled,
        density_plane,
        density_line,
        sigma_feature_cuda,
        plane_mul_line_list,
    ):
        """
        compute density feature with custom kernel.
        """
        for idx_plane in range(len(density_plane)):
            # torch.cuda.synchronize()
            # starter = torch.cuda.Event(enable_timing=True); starter.record();
            # compute_appfeature.gridsample_sum(xyz_sampled,
            #                                   density_plane[idx_plane], density_line[idx_plane],
            #                                   plane_mul_line_list)
            compute_appfeature.gridsample_sum_xyz_b(
                xyz_sampled,
                b_sampled,
                density_plane[idx_plane],
                density_line[idx_plane],
                plane_mul_line_list,
            )

            if idx_plane == 0:
                sigma_feature_cuda[:] = plane_mul_line_list
            else:
                sigma_feature_cuda[:] = sigma_feature_cuda + plane_mul_line_list

        return sigma_feature_cuda

    # by XHR
    # xyb_bz: 将xyz与b分离, 单独传入b_tensor, 其单元类型是int
    def compute_appfeature_multi_xyb_bz_cuda(
        self,
        app_plane,
        app_line,
        basis_mat_weight,
        xyz_sampled,
        b_sampled,
        plane_mul_line_list,
        app_feature_cuda,
    ):
        """
        compute appfeature with custom kernel.
        """
        grid_channel = app_plane[0].shape[1]

        # uses_time = 0.0
        # warmup_num = 0
        test_num = 1
        for _ in range(test_num):
            compute_appfeature.gridsample_ew_xyb_bz(
                xyz_sampled, b_sampled, app_plane, app_line, plane_mul_line_list, grid_channel, 0
            )
            global temp  # pylint: disable=W0601
            temp = plane_mul_line_list
            plane_mul_line_T = plane_mul_line_list.view(-1, xyz_sampled.shape[0])
            gemm_gac.run(
                input=plane_mul_line_T,
                weight=basis_mat_weight.to(xyz_sampled.device),
                output=app_feature_cuda,
                relu=False,
            )

        return app_feature_cuda

    def forward(self, rays_chunk, app_code=None, white_bg=True):
        """
        multi blocks forward with custom cuda kernels.
        """
        distance_scale = self._tensorf.distance_scale
        N_rays = rays_chunk.shape[0]
        z_vals_cuda = self._tensorf.z_vals_cuda
        beta = self._tensorf.beta
        weight_cuda = self._tensorf.weight_cuda

        with torch.no_grad():  # with torch.cuda.device(tensorf.device):
            ret_cuda_0 = self.tensorf_multiBlockParallel_cuda_part0(rays_chunk)

            self._tensorf.xyz_sampled_top = ret_cuda_0["xyz_sampled_cuda"]
            z_vals_cuda = ret_cuda_0["z_vals_cuda"]
            valid_b_sampled = ret_cuda_0["valid_b_sampled"]
            alpha_mask = ret_cuda_0["ray_valid"]
            valid_samples_num = self._tensorf.xyz_sampled_top.size(1)  # d

            b_sampled_alpha_mask = valid_b_sampled[alpha_mask]
            xyz_sampled_alpha_mask = self._tensorf.xyz_sampled_top[alpha_mask]  # xyz_sample_valid[alpha_mask]

            # ## by XHR
            # sigma_feature = torch.zeros((xyz_sampled_alpha_mask.shape[0],), device=xyz_sampled_alpha_mask.device)
            sigma_feature = self._tensorf.density_feature_cuda.view(-1)[: xyz_sampled_alpha_mask.shape[0]].view(
                xyz_sampled_alpha_mask.shape[0]
            )

            plane_mul_line_list = self._tensorf.plane_mul_line_list.view(-1)[: xyz_sampled_alpha_mask.shape[0]].view(
                xyz_sampled_alpha_mask.shape[0]
            )
            sigma_feature = self.compute_densityfeature_xyzb_cuda(
                xyz_sampled_alpha_mask,
                b_sampled_alpha_mask,
                self._tensorf.density_plane,
                self._tensorf.density_line,
                sigma_feature,
                plane_mul_line_list,
            )

            validsigma = self._tensorf.feature2density(sigma_feature)
            sigma_feature_cuda_cut = self._tensorf.sigma_feature_cuda.view(-1)[: N_rays * valid_samples_num].view(
                N_rays, valid_samples_num
            )
            sigma_feature_cuda_cut = torch.zeros_like(sigma_feature_cuda_cut)  # 全部置为零
            sigma_feature_cuda_cut[alpha_mask] = validsigma

            z_vals_cuda_cut = z_vals_cuda.view(-1)[: N_rays * valid_samples_num].view(N_rays, valid_samples_num)

            sigma_cuda = torch.nn.functional.relu(sigma_feature_cuda_cut)
            sigma_cuda = sigma_cuda.view(N_rays, -1)
            compute_beta.cuda(sigma_cuda, z_vals_cuda_cut, beta, distance_scale)

            beta_cut = beta.view(-1)[: N_rays * valid_samples_num].view(N_rays, valid_samples_num)
            compute_weight.cuda(beta_cut, weight_cuda)

            # N_rays = rays_chunk.size(0)
            self._tensorf.z_vals_top = (
                z_vals_cuda_cut  # z_vals.view(-1)[: N_rays * valid_samples_num].view(N_rays, valid_samples_num)
            )
            self._tensorf.weight_top = weight_cuda.view(-1)[: N_rays * valid_samples_num].view(
                N_rays, valid_samples_num
            )
            # tensorf_part2_cuda(tensorf)

            # # D2D copy
            # typecst = 0
            # app_mask: [Nrays, valid_samples_num]
            self._tensorf.app_mask = self._tensorf.weight_top > self._tensorf.rayMarch_weight_thres

            # by XHR
            # app_mask: [90w, 3]
            xyz_sampled = self._tensorf.xyz_sampled_top[
                self._tensorf.app_mask
            ]  # xyzb_sampled[tensorf.app_mask][:,:3].clone()
            b_sampled = valid_b_sampled[self._tensorf.app_mask]

            # appfeature_cuda = [90w, 27]
            basis_mat_weight = self._tensorf.basis_mat.weight
            appfeature_cuda = self._tensorf.appfeature_cuda.view(-1)[
                : xyz_sampled.shape[0] * basis_mat_weight.shape[0]
            ].view(xyz_sampled.shape[0], basis_mat_weight.shape[0])

            # 2D:xy, 1.5D:bz ---->  2.5D:xyb, 1.5D:bz
            plane_mul_line_list = self._tensorf.plane_mul_line_list.view(-1)[
                : len(self._tensorf.app_plane) * (self._tensorf.app_plane[0].shape[1] * xyz_sampled.shape[0])
            ].view(len(self._tensorf.app_plane), (self._tensorf.app_plane[0].shape[1] * xyz_sampled.shape[0]))
            appfeature_cuda = self.compute_appfeature_multi_xyb_bz_cuda(
                self._tensorf.app_plane,
                self._tensorf.app_line,
                basis_mat_weight,
                xyz_sampled,
                b_sampled,
                plane_mul_line_list,
                appfeature_cuda,
            )

            fake_xyzb_sampled_idxs = torch.zeros(xyz_sampled.shape[:-1], dtype=torch.long, device=xyz_sampled.device)
            if app_code is not None:
                fake_xyzb_sampled_idxs = fake_xyzb_sampled_idxs * app_code.long()
            app_latent = self._tensorf.embedding_app(fake_xyzb_sampled_idxs)  # [tensorf.app_mask])

            # feape = viewpe = 2
            viewdirs = rays_chunk[:, 3:6].view(-1, 1, 3).expand(self._tensorf.xyz_sampled_top.shape)  # cpu op
            # To improve 1 : 3.7ms, viewdirs [Nrays, 50, 3] ---> [9340943, 3]
            viewdirs = viewdirs[
                self._tensorf.app_mask
            ]  # aten:nonzero, gpu op: DeviceReduceKernel + DeviceReduceSingleTileKernel + Memcpy D2H + ...

            #
            pc_cuda = self._tensorf.pc_cuda.view(-1)[: xyz_sampled.shape[0] * 152].view(xyz_sampled.shape[0], 152)
            pc.pe_concate(
                appfeature_cuda, self._tensorf.renderModule.feape, viewdirs, self._tensorf.renderModule.viewpe, pc_cuda
            )  # Already improve 2 : no return, half + pad: 10.6ms(first!!!!!!!!!!!!!!!!!!!!)

            # mlp_in: [9340943, 150] ---> [9340943, 152]
            # Concatenate in 2nd dimension
            mlp_in = torch.cat([app_latent.half(), pc_cuda], dim=1)
            # print(mlp_in.shape)
            # => torch.Size([128, 4, 150, 150])

            # mlp_layer1_out_top: [9340943, 128] ---> [9340943, 128]
            m = pc_cuda.size(0)
            n1 = self._tensorf.layer3_weight_paded.size(1)
            # n2 = tensorf.layer3_weight_paded.size(1)
            # k2 = 128;
            n3 = self._tensorf.layer3_weight_paded.size(0)
            mlp_layer1_out_top = self._tensorf.mlp_layer1_out.view(-1)[: m * n1].view(m, n1)
            # mlp_layer2_out_top = tensorf.mlp_layer1_out.view(-1)[: m * n2].view(m, n2)
            mlp_layer3_out_top = self._tensorf.mlp_layer1_out.view(-1)[: m * n3].view(m, n3)

            # layer1~3: mlp 0,2,4
            # relu: mlp 1,3
            gemm.run(
                mlp_in,
                self._tensorf.layer1_weight_paded,
                mlp_layer1_out_top,
                relu=True,  # tensorf.renderModule.mlp[0].weight, #
            )
            gemm.run(mlp_layer1_out_top, self._tensorf.renderModule.mlp[2].weight.half(), mlp_layer1_out_top, relu=True)
            gemm.run(
                mlp_layer1_out_top,
                self._tensorf.layer3_weight_paded,  # layer3.weight, #layer3_weight_paded, #
                # layer3.weight, mlp_layer2_out,
                mlp_layer3_out_top,
                relu=False,
            )
            # sigmoid : 0.133ms
            mlp_layer3_out_top = torch.sigmoid(mlp_layer3_out_top)

            # rgb: [chunk_cuda, 50, 3]
            rgb = self._tensorf.rgb.view(-1)[: N_rays * valid_samples_num * 3].view(N_rays, valid_samples_num, 3)
            # aten:index : 1.38ms
            rgb[self._tensorf.app_mask] = mlp_layer3_out_top[:, :3]
            # mul+reduce : 1.481 + 0.774ms
            rgb_map = torch.sum(self._tensorf.weight_top[..., None] * rgb, -2)

            # white background
            if white_bg:
                acc_map = torch.sum(self._tensorf.weight_top, -1)
                rgb_map = rgb_map + (1.0 - acc_map[..., None])

        outputs = {
            "xyz_sampled_cuda": self._tensorf.xyz_sampled_top,  # xyz_sampled_cuda,
            "z_vals_cuda": self._tensorf.z_vals_top,  # z_vals_cuda,
            "alpha_mask": alpha_mask,
            "sigma_feature_cuda": sigma_feature,
            "sigma_cuda": sigma_cuda,
            "weight_cuda": self._tensorf.weight_top,
            "app_mask_cuda": self._tensorf.app_mask,
            "appfeature_cuda": appfeature_cuda,
            "pc_cuda": self._tensorf.pc_cuda,
            # above will update every chunk
            "mlp_layer3_out_top": mlp_layer3_out_top,  # differ across chunk
            "rgb_map": rgb_map,  # differ across chunk
        }
        return outputs


class TPMultiBlockParallelCuda(MultiBlockParallelCuda):
    """
    tensorf_multiBlockParallel_cuda_v0606_tp
    """

    def __init__(self) -> None:
        super().__init__()
        # self._tensorf.fullgrid = 0
        # 0 for v0606
        # 1 for v0607
        # self._tensorf.column_major = 0

    def allocate_tensors(self, tensorf, N_rays):
        """
        allocate cuda tensors in advance with tp
        """
        super().allocate_tensors(tensorf, N_rays)

        basis_mat_weight = self._tensorf.basis_mat.weight  # basis_mat_weight.shape[0] = 144
        if (
            hasattr(self._tensorf, "tensor_parallel_group_world_size")
            and self._tensorf.tensor_parallel_group_world_size > 0
        ):
            self._tensorf.basis_mat_weight_tp = (
                basis_mat_weight.reshape(
                    basis_mat_weight.shape[0],
                    -1,
                    self._tensorf.tensor_parallel_group_world_size,
                    len(self._tensorf.app_plane),
                )
                .transpose(2, 3)
                .contiguous()  # (0, 1)
                # .flatten(0, 2)
            )

    def compute_densityfeature_xyzb_cuda(
        self,
        xyz_sampled,
        b_sampled,
        density_plane,
        density_line,
        sigma_feature_cuda,
        plane_mul_line_list,
    ):
        """
        compute densityfeature with tp.
        """
        sigma_feature_cuda = super().compute_densityfeature_xyzb_cuda(
            xyz_sampled,
            b_sampled,
            density_plane,
            density_line,
            sigma_feature_cuda,
            plane_mul_line_list,
        )

        # 1st all_gather: Tp from ljun
        sigma_feature_cuda = sigma_feature_cuda.view(1, -1)
        all_sigma_feature = self._tensorf.tensor_parallel_all_gather_func(sigma_feature_cuda)
        sigma_feature_cuda = torch.sum(all_sigma_feature, dim=0)
        return sigma_feature_cuda

    def compute_appfeature_multi_xyb_bz_cuda(
        self, app_plane, app_line, basis_mat_weight, xyz_sampled, b_sampled, plane_mul_line_list, app_feature_cuda
    ):
        """
        compute app feature with tp.
        """
        # app_plane, app_line, basis_mat_weight
        # tensorf.basis_mat_weight_tp
        # basis_mat_weight = self._tensorf.basis_mat.weight

        grid_channel = app_plane[0].shape[1]

        # uses_time = 0.0
        # warmup_num = 0
        test_num = 1
        for _ in range(test_num):
            compute_appfeature.gridsample_ew_xyb_bz(
                xyz_sampled, b_sampled, app_plane, app_line, plane_mul_line_list, grid_channel, 0
            )
            # global temp
            plane_mul_line_T = plane_mul_line_list.view(-1, xyz_sampled.shape[0])
            local_rank = self._tensorf.tensor_parallel_local_rank
            groupsize = (int)(48 / self._tensorf.tensor_parallel_group_world_size)
            # local_rank * 24 : (local_rank+1) * 24

            gemm_gac.run(
                input=plane_mul_line_T,  # plane_mul_line_T: [3,48,m]
                # input = all_plane_mul_line_T, # all_plane_mul_line_T: [Ntp, 3, 48/Ntp, m]
                weight=basis_mat_weight.view(27, 3, 48)[
                    :, :, local_rank * groupsize : (local_rank + 1) * groupsize  # 24:
                ]
                .contiguous()
                .to(xyz_sampled.device),
                output=app_feature_cuda,
                relu=False,
            )

        app_feature_cuda = app_feature_cuda.view(1, -1, app_feature_cuda.shape[-1])
        all_app_feature_cuda = self._tensorf.tensor_parallel_all_gather_func(app_feature_cuda)
        # all_app_feature_cuda = all_app_feature_cuda.transpose(0, 1).flatten(1, 2)
        app_feature_cuda = torch.sum(all_app_feature_cuda, dim=0)
        return app_feature_cuda

# pylint: disable=E1102
import math
import os
import threading

import numpy as np
import torch

from dist_render.comm.communication import broadcast
from dist_render.comm.dynamic_loader import should_load_full_plane
from dist_render.comm.parallel_context import ParallelGroup

from .alpha_mask import AlphaGridMask
from .gridnerf_elastic_cuda_base import DistRenderGridNeRFElasticCudaBase


class DistRenderGridNeRFElasticCuda(DistRenderGridNeRFElasticCudaBase):
    """
    class for moving area version of TensorVMSplit_multi

    Note:
        One area may have multiple block.
    """

    def __init__(self, aabb, gridSize, device, **kargs):
        self.density_plane = None
        self.app_plane = None
        self.density_plane_all = None
        self.app_plane_all = None
        self.density_line_all = None
        self.app_line_all = None

        super().__init__(aabb, gridSize, device, **kargs)

        self.load_offset = None
        self.select_offset = None

        self.preload_lock = threading.Lock()
        self.lock_by_render_thread = False

        self.d2d_stream = torch.cuda.Stream()
        self.inited = False

    def allocate_buffers(self, N_rays):
        m = N_rays * 90
        self.density_plane = self.buffers[self.current_buffer_idx].density_plane
        self.density_line = self.buffers[self.current_buffer_idx].density_line
        self.app_plane = self.buffers[self.current_buffer_idx].app_plane
        self.app_line = self.buffers[self.current_buffer_idx].app_line

        if not self.inited:
            self.inited = True

            self.hw_in = torch.tensor(
                [
                    [
                        self.density_plane[i].shape[2],
                        self.density_plane[i].shape[3],
                        self.density_line[i].shape[2],
                        self.density_line[i].shape[3],
                    ]
                    for i in range(len(self.density_plane))
                ],
                device="cuda",
                dtype=torch.int,
            )

            self.grid_channel = self.app_plane[0].shape[4]
            self.plane_mul_line_list = torch.zeros(
                (m, len(self.app_plane), self.grid_channel), device="cuda", dtype=torch.half
            )

            self.plane_line_ptr = torch.empty((len(self.density_plane) * 2), device="cuda", dtype=torch.int)

    def allocate_tensors(self, N_rays):
        Max_samples_num = self.n_importance

        # sampleray
        self.xyz_sampled_cuda = torch.empty((N_rays, Max_samples_num, 3), device="cuda", dtype=torch.float)
        self.z_vals_cuda = torch.empty((N_rays, Max_samples_num), device="cuda", dtype=torch.float)
        self.maskbbox_cuda = torch.randint(2, [N_rays, Max_samples_num]).cuda() == 0

        self.tvals_min_cuda = torch.rand((math.ceil(N_rays / 1024)), device="cuda", dtype=torch.float)
        self.tvals_min_index_cuda = torch.ones(1, device="cuda", dtype=torch.float)
        self.sigma_feature_bool_cuda = (
            torch.randint(  # randbooltrue(
                2,
                [
                    1,
                    1,
                    N_rays,
                    Max_samples_num,
                    1,
                ],
            ).cuda()
            == 0
        )
        self.density_plane_line_sum = self.alphaMask.alpha_volume
        self.valid_samples_num = torch.from_numpy(np.array([-1]))

        self.column_major = 1  # 0 for v0606
        # 1 for v0607

        # for pe and mlp
        m = N_rays * 90
        mlp_dtype_ = torch.half
        self.pc_cuda = torch.zeros((m, 152), device="cuda").half()
        # encoding and expand index place
        self.fea_offset = 0
        self.view_offset = 27  # 39 #
        self.view_encoding_offset = self.view_offset + 3  # 27 #
        self.feature_encoding_offset = 42

        # shape: [128, 198]
        weight_half = self.renderModule.mlp[0].weight.half()
        encappdim = self.embedding_app.embedding_dim

        self.layer1_weight_paded = torch.cat(
            [weight_half[:, 27 : 27 + encappdim], weight_half[:, :27], weight_half[:, 27 + encappdim :]], dim=1
        )
        self.layer1_weight_paded = torch.nn.functional.pad(self.layer1_weight_paded, [0, 2, 0, 0])

        self.layer3_weight_paded = torch.nn.functional.pad(self.renderModule.mlp[4].weight.half(), [0, 0, 0, 5])

        basis_mat_weight = self.basis_mat.weight  # basis_mat_weight.shape[0] = 144

        # AssignBlocksToSamples & compute_densityfeature
        self.b_sampled_cuda = torch.zeros((N_rays, Max_samples_num), device="cuda", dtype=torch.int)

        self.sigma_feature_valid_cuda = torch.zeros((m), device="cuda", dtype=torch.float32)

        # compute_weight
        self.beta = torch.empty((N_rays, Max_samples_num), device="cuda", dtype=torch.float)
        self.weight_cuda = torch.empty((N_rays, Max_samples_num), device="cuda", dtype=torch.float)
        self.sigma_feature_cuda = torch.rand(
            1, 1, N_rays, Max_samples_num, 1  # density_plane_line_sum.shape[1],  # int(N_samples*(1-tvals_min_cuda[0]))
        ).cuda()

        # by SZL
        self.pc_cuda = torch.empty((m, 152), device="cuda").half()

        # k1 = 152; # if k1 = 150
        n1 = self.layer3_weight_paded.size(1)
        # n2 = self._self.layer3_weight_paded.size(1)
        # k2 = 128;
        # n2 = 128; # k3 = 128;
        self.mlp_layer1_out = torch.empty((m, n1), dtype=mlp_dtype_, device="cuda")

        n3 = self.layer3_weight_paded.size(0)
        self.mlp_layer3_out = torch.empty((m, n3), dtype=mlp_dtype_, device="cuda")
        # rgb: [chunk_cuda, 50, 3]

        self.rgb = torch.zeros((N_rays, Max_samples_num, 3), dtype=mlp_dtype_, device="cuda")

        # encoding and expand index place
        self.fea_offset = 0
        self.view_offset = 27  # 39 #
        self.view_encoding_offset = self.view_offset + 3  # 27 #
        self.feature_encoding_offset = 42

        if self.args.encode_app:
            # shape: [128, 198]
            weight_half = self.renderModule.mlp[0].weight.half()
            encappdim = self.embedding_app.embedding_dim
            if self.column_major:
                self.layer1_weight_paded = torch.cat(
                    [
                        weight_half[:, 27 : 27 + encappdim],
                        weight_half[:, :27],
                        weight_half[:, 27 + encappdim : 27 + encappdim + 3],
                        weight_half[:, 27 + 48 + 3 + 108 : 27 + 48 + 3 + 108 + 12],
                        weight_half[:, 27 + 48 + 3 : 27 + 48 + 3 + 108],
                    ],
                    dim=1,
                )
                self.layer1_weight_paded = torch.nn.functional.pad(self.layer1_weight_paded, [0, 2, 0, 0])
            else:
                self.layer1_weight_paded = torch.cat(
                    [weight_half[:, 27 : 27 + encappdim], weight_half[:, :27], weight_half[:, 27 + encappdim :]], dim=1
                )
                self.layer1_weight_paded = torch.nn.functional.pad(self.layer1_weight_paded, [0, 2, 0, 0])
        else:
            if self.column_major:
                weight_half = self.renderModule.mlp[0].weight.half()
                self.layer1_weight_paded = torch.cat(
                    [
                        weight_half[:, : self.view_encoding_offset],
                        weight_half[:, self.view_encoding_offset + 108 : self.view_encoding_offset + 108 + 12],
                        weight_half[:, self.view_encoding_offset : self.view_encoding_offset + 108],
                    ],
                    dim=1,
                )
                print(self.layer1_weight_paded.shape)
                self.layer1_weight_paded = torch.nn.functional.pad(self.layer1_weight_paded, [0, 2, 0, 0])
            else:
                self.layer1_weight_paded = torch.nn.functional.pad(self.renderModule.mlp[0].weight.half(), [0, 2, 0, 0])
        # layer3_weight_paded: [3, 128] ---> [8, 128]
        self.layer3_weight_paded = torch.nn.functional.pad(self.renderModule.mlp[4].weight.half(), [0, 0, 0, 5])

        basis_mat_weight = self.basis_mat.weight
        # Creat mlp arguments for self
        self.m = N_rays * 16
        self.k1 = self.layer1_weight_paded.shape[1]
        self.n1 = self.layer1_weight_paded.shape[0]
        self.k2 = self.renderModule.mlp[2].weight.shape[1]
        self.n2 = self.renderModule.mlp[2].weight.shape[0]
        self.k3 = self.layer3_weight_paded.shape[1]
        self.n3 = self.layer3_weight_paded.shape[0]
        self.gemm_app_k = self.basis_mat.weight.shape[1]
        self.gemm_app_n = self.basis_mat.weight.shape[0]

        if self.column_major:
            self.index_ori = torch.arange(0, N_rays).int().to(self.device).reshape(N_rays, 1)
            self.appfeature_cuda_half = torch.empty(
                (m, basis_mat_weight.shape[0]), dtype=torch.half, device=self.device
            )
        else:
            self.appfeature_cuda = torch.empty((m, basis_mat_weight.shape[0]), device=self.device)

        self.pipeline = self.column_major  # to verify
        app_latent_shape = 1 if self.pipeline else m  # xyz_sampled.shape[:-1]
        self.fake_xyzb_sampled_idxs = torch.zeros(app_latent_shape, dtype=torch.long, device="cuda")

        if self.pipeline:

            pipeline_tmp_m = 256 * 108 * 20 * 10
            self.pipeline_tmp_m = pipeline_tmp_m - pipeline_tmp_m % 8
            self.mlp_in_tmp = torch.zeros(self.pipeline_tmp_m * self.k1, dtype=mlp_dtype_).to(self.device)
            self.mlp_layerout_tmp = torch.zeros(self.pipeline_tmp_m, self.n1, dtype=mlp_dtype_).to(self.device)

        else:
            # by SZL
            self.mlp_in = torch.empty(m * (self.k1 + basis_mat_weight.shape[0]), device=self.device).half()

        self.mlp_layer3_out = torch.empty((m, self.n3), dtype=mlp_dtype_).to(self.device)

        # rgb: [chunk_cuda, 50, 3]
        self.rgb = torch.ones((N_rays, Max_samples_num, 3), dtype=mlp_dtype_).to(self.device)

    def update_device(self, device):
        """
        DESC:
            put some args to device, while density/app plane remain on cpu.
        """
        self.device = device
        self.renderModule = self.renderModule.to(device)
        if self.run_nerf:
            self.nerf = self.nerf.to(device)
        self.gridSize = self.gridSize.to(device)
        self.density_line = self.density_line.to(device)
        self.app_line = self.app_line.to(device)
        self.basis_mat = self.basis_mat.to(device)
        if self.nonlinear_density:
            self.basis_den = self.basis_den.to(device)
        if self.args.encode_app:
            self.embedding_app = self.embedding_app.to(device)

        if self.alphaMask is not None:
            # for rank other to update device successfully.
            self.alphaMask.update_device(device)

    def create_alpha_mask(self, ckpt):
        if "alphaMask.aabb" in ckpt.keys():
            length = np.prod(ckpt["alphaMask.shape"])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt["alphaMask.mask"])[:length].reshape(ckpt["alphaMask.shape"])
            )
            self.alphaMask = AlphaGridMask(
                self.device,
                ckpt["alphaMask.aabb"],
                alpha_volume.float().to(self.device),
            )

    def permute_and_split_model(self):
        """
        Desc:
            Permute the model plane from NCDHW to NDHWC. The reason is that
            we need to ensure every block is memory contiguous in the channel dimension,
            so we can merge some nccl/h2d/d2d operation into one to reduce the launch cost.
        """
        self.density_plane_all = [None for _ in range(len(self.density_plane))]
        self.app_plane_all = [None for _ in range(len(self.app_plane))]

        for idx in range(len(self.density_plane)):
            self.density_plane[idx] = self.density_plane[idx].permute(0, 2, 3, 4, 1)  # pylint: disable=E1136,E1137
            self.app_plane[idx] = self.app_plane[idx].permute(0, 2, 3, 4, 1)  # pylint: disable=E1136,E1137
            self.density_plane_all[idx] = list(
                torch.chunk(
                    self.density_plane[idx], chunks=self.density_plane[idx].shape[1], dim=1  # pylint: disable=E1136
                )
            )
            self.app_plane_all[idx] = list(
                torch.chunk(self.app_plane[idx], chunks=self.app_plane[idx].shape[1], dim=1)  # pylint: disable=E1136
            )

            for i in range(len(self.density_plane_all[idx])):
                self.density_plane_all[idx][i] = self.density_plane_all[idx][i].squeeze(1)
                self.app_plane_all[idx][i] = self.app_plane_all[idx][i].squeeze(1)
                self.density_plane_all[idx][i] = self.density_plane_all[idx][i].contiguous().pin_memory()
                self.app_plane_all[idx][i] = self.app_plane_all[idx][i].contiguous().pin_memory()

        self.density_line_all = self.density_line
        self.app_line_all = self.app_line

    def load(self, ckpt):
        """
        Desc:
            load checkpoint

        Args:
            ckpt (dict): checkpoint
        """
        self.create_alpha_mask(ckpt)
        self.load_state_dict(ckpt["state_dict"], strict=False)

    def calculate_load_offset(self):
        """
        Desc:
            Calculate the threshold to trigger loading thread
            when the camera pose minus buffer's plane center is
            out of load threshold.
        """
        if self.load_offset is None:
            plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]
            offset_ratio = 0.51
            self.load_offset = torch.cuda.FloatTensor([offset_ratio * plane_width, offset_ratio * plane_height])

        return self.load_offset

    def calculate_select_offset(self):
        """
        Desc:
            Calculate the threshold to choose which direction to load.
        """
        if self.select_offset is None:
            plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]
            offset_ratio = 0.5 - (0.51 - 0.5)
            self.select_offset = torch.cuda.FloatTensor([offset_ratio * plane_width, offset_ratio * plane_height])

        return self.select_offset

    def calculate_distance(self, pose_o):
        """
        Desc:
            Calculate the distance between given camera pose and buffer's plane center,
            which is used to compare with `load_offset` and `select_offset`.

        Args:
            pose_o (torch.Tensor): camera pose contains xyz coordination.
        """
        # assert pose as [x, y, z]
        return torch.abs(pose_o[:2] - self.buffers[self.current_buffer_idx].center)

    def meet_load_threshold(self, pose_o):
        """
        DESC:
            Check whether it should be loaded.

        Args:
            pose_o (torch.Tensor): camera pose contains xyz coordination.
        """
        if self.load_offset is None:
            self.calculate_load_offset()
        normalized_pose_o = self.normalize_coord(pose_o[:3])
        distance = self.calculate_distance(normalized_pose_o)
        return (distance > self.load_offset).any().item()

    def switch_buffers(self):
        """
        DESC:
            Change the buffer to use in rendering.

        Args:
            pose_o (torch.Tensor): camera pose contains xyz coordination.
        """
        self.current_buffer_idx ^= 1
        return self.current_buffer_idx

    def calculate_block_idx(self, pose_o):
        """
        Desc:
            Find out which block contain the pose.

        Args:
            pose_o (torch.Tensor): camera pose contains xyz coordination.
        """
        plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]
        normalized_pose_o = self.normalize_coord(pose_o[:3])
        normalized_pose_o[:2] += 1
        idx_x = torch.floor(normalized_pose_o[0] / plane_width)
        idx_y = torch.floor(normalized_pose_o[1] / plane_height)
        block_idx = idx_x * self.args.plane_division[1] + idx_y
        normalized_pose_o[:2] -= 1

        return block_idx.int()

    def calculate_block_center(self, block_idx, device="cpu"):
        """
        Desc:
            Find out the buffer's plane center of the local plane,
            whose center block's index is `block_idx`.

        Args:
            block_idx (torch.Tensor): Central block's index of local plane.
        """
        idx_x = block_idx // self.args.plane_division[1]
        idx_y = block_idx % self.args.plane_division[1]
        plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]

        if self.neighbour_width % 2 == 1:
            block_center = torch.cuda.FloatTensor(
                [-1 + (0.5 + idx_x) * plane_width, -1 + (0.5 + idx_y) * plane_height], device=device
            )
        else:
            block_center = torch.cuda.FloatTensor(
                [-1 + (0 + idx_x) * plane_width, -1 + (0 + idx_y) * plane_height], device=device
            )

        return block_center

    def calculate_block_neighbours(self, block_idx):
        """
        Desc:
            Find out all blocks' index within local plane,
            whose central block's index is `block_idx`.

        Args:
            block_idx (torch.Tensor): Central block's index of local plane.

        NOTE:
            neighbours include `block_idx` itself.
        """
        idx_x = block_idx // self.args.plane_division[1]
        idx_y = block_idx % self.args.plane_division[1]
        neighbours = []
        radius = self.neighbour_width // 2
        rbound = 0
        if self.neighbour_width % 2 == 0:
            rbound = radius
        else:
            rbound = radius + 1

        for offset_x in range(-radius, rbound):
            for offset_y in range(-radius, rbound):

                idx_x_ = idx_x + offset_x
                idx_y_ = idx_y + offset_y
                if (
                    idx_x_ < 0
                    or idx_y_ < 0
                    or idx_x_ >= self.args.plane_division[0]
                    or idx_y_ >= self.args.plane_division[1]
                ):
                    neighbours.append(0)
                else:
                    neighbour_idx = idx_x_ * self.args.plane_division[1] + idx_y_
                    neighbours.append(int(neighbour_idx.item()))
        return neighbours

    def calculate_next_block_idx(self, pose_o):
        """
        Desc:
            Predict the central block's index to be used in the next time using
            `load_offset` and `select_offset`.

        Args:
            pose_o (torch.Tensor): camera pose contains xyz coordination.
        """
        center = self.buffers[self.current_buffer_idx].center
        center_block_idx = self.buffers[self.current_buffer_idx].center_block_idx

        idx_x = center_block_idx // self.args.plane_division[1]
        idx_y = center_block_idx % self.args.plane_division[1]

        normalized_pose_o = self.normalize_coord(pose_o[:3])
        x_offset = normalized_pose_o[0] - center[0]
        y_offset = normalized_pose_o[1] - center[1]

        next_block_idx_x = idx_x
        next_block_idx_y = idx_y

        x_idx_offset = 0
        y_idx_offset = 0

        # too far away check
        plane_width, plane_height = 2 / self.args.plane_division[0], 2 / self.args.plane_division[1]
        if torch.ge(torch.abs(x_offset), (self.neighbour_width * plane_width)) or torch.ge(
            torch.abs(y_offset), (self.neighbour_width * plane_height)
        ):
            next_block_idx_x = torch.floor((normalized_pose_o[0] + 1) / plane_width)
            next_block_idx_y = torch.floor((normalized_pose_o[1] + 1) / plane_height)
            next_block_idx_x = max(next_block_idx_x, 0)
            next_block_idx_x = min(next_block_idx_x, self.args.plane_division[0] - 1)
            next_block_idx_y = max(next_block_idx_y, 0)
            next_block_idx_y = min(next_block_idx_y, self.args.plane_division[1])

            next_block_idx = next_block_idx_x * self.args.plane_division[1] + next_block_idx_y
            next_block_idx = next_block_idx.int()
            return next_block_idx

        if self.select_offset is None:
            self.calculate_select_offset()

        if x_offset >= self.load_offset[0]:
            x_idx_offset = 1
            if y_offset >= self.select_offset[1]:
                y_idx_offset = 1
            elif y_offset <= -self.select_offset[1]:
                y_idx_offset = -1
        elif x_offset <= -self.load_offset[0]:
            x_idx_offset = -1
            if y_offset >= self.select_offset[1]:
                y_idx_offset = 1
            elif y_offset <= -self.select_offset[1]:
                y_idx_offset = -1

        if y_offset >= self.load_offset[1]:
            y_idx_offset = 1
            if x_offset >= self.select_offset[0]:
                x_idx_offset = 1
            elif x_offset <= -self.select_offset[0]:
                x_idx_offset = -1
        elif y_offset <= -self.load_offset[1]:
            y_idx_offset = -1
            if x_offset >= self.select_offset[0]:
                x_idx_offset = 1
            elif x_offset <= -self.select_offset[0]:
                x_idx_offset = -1

        next_block_idx_x += x_idx_offset
        next_block_idx_y += y_idx_offset
        if next_block_idx_x < 0 or next_block_idx_x >= self.args.plane_division[0]:
            next_block_idx_x -= x_idx_offset
        if next_block_idx_y < 0 or next_block_idx_y >= self.args.plane_division[1]:
            next_block_idx_y -= y_idx_offset

        next_block_idx = next_block_idx_x * self.args.plane_division[1] + next_block_idx_y
        return next_block_idx

    def create_new_plane_stack(self, neighbours):
        """
        Desc:
            Create app/density plane's memory space on device for buffers and
            update the plane using given neighbours' plane value.

        Args:
            neighbours (list[int]): Specify the block to be used.
        """
        density_blocks, app_blocks = [], []
        for idx_plane in range(len(self.density_plane_all)):
            density_block_list = [self.density_plane_all[idx_plane][neighbour] for neighbour in neighbours]
            density_block = torch.stack(density_block_list, dim=1)
            app_block_list = [self.app_plane_all[idx_plane][neighbour] for neighbour in neighbours]
            app_block = torch.stack(app_block_list, dim=1)

            density_blocks.append(density_block)
            app_blocks.append(app_block)
        density_plane_blocks, app_plane_blocks = torch.nn.ParameterList(density_blocks), torch.nn.ParameterList(
            app_blocks
        )
        return density_plane_blocks, app_plane_blocks

    def create_new_line_stack(self, neighbours):
        """
        Desc:
            Create app/density line's memory for buffers.

        Args:
            neighbours (list[int]): Specify the block to be used.
        """
        density_blocks, app_blocks = [], []
        for idx_line in range(len(self.density_line_all)):
            density_block = self.density_line_all[idx_line][..., neighbours].detach()
            app_block = self.app_line_all[idx_line][..., neighbours].detach()
            density_blocks.append(density_block)
            app_blocks.append(app_block)
        return torch.nn.ParameterList(density_blocks), torch.nn.ParameterList(app_blocks)

    def update_plane_buffer(self, buffer_idx, neighbours):
        """
        Desc:
            Using given neighbours' block to update the buffer's value.

        Args:
            buffer_idx (int): the buffer index within buffers, specific which buffer to be updated.
            neighbours (list[int]): Specify the block to be used to update the buffer.
        """
        density_plane_blocks, app_plane_blocks = self.create_new_plane_stack(neighbours)
        for i in range(len(self.buffers[buffer_idx].density_plane)):
            self.buffers[buffer_idx].density_plane[i] = density_plane_blocks[i].to(self.args.device, non_blocking=True)
            self.buffers[buffer_idx].app_plane[i] = app_plane_blocks[i].to(self.args.device, non_blocking=True)

    def calculate_list_diff(self, list_a, list_b):
        """
        Desc:
            Simple implement to calculate the similarities and differences between `list_a` and `list_b`.

        Args:
            list_a (list[int]): one list to be used.
            list_b (list[int]): another list to be used.
        """
        if list_b is None:
            return list_a

        diff_list, same_list = [], []
        for item in list_a:
            if item not in list_b:
                diff_list.append(int(item))
            else:
                same_list.append(int(item))

        return diff_list, same_list

    @torch.no_grad()
    def calculate_h2d(self, buffer_idx, neighbours, neighbours_to_load, nccl_only=False):
        """
        Desc:
            Using PCI-e and NVLink to load the block from cpu to gpu.
            First using PCI-e to load plane param from cpu to gpu on one rank;
            Then broadcast the param from one gpu to other gpu using NVLink;
            All the operations are asynchronous to ensure loading thread won't block rendering thread
            either on CPU or on GPU.

        Args:
            buffer_idx (int): the buffer index within buffers, specific which buffer to be updated.
            neighbours (list[int]): Specify all blocks to be used to update the buffer.
                If block already on device, it will be used in d2d thread, or it will be used in h2d thread.
            neighbours_to_load (list[int]): Specify the block to be used in h2d thread.
            nccl_only (bool): used to distinguish whether to do the h2d operation. If it's True, only do
                the nccl operations.
        """
        handles = []
        for new_block_idx, neighbour in enumerate(neighbours):
            if neighbour in neighbours_to_load:
                for idx_plane in range(len(self.density_plane_all)):
                    if not nccl_only:
                        self.buffers[buffer_idx].density_plane[idx_plane][
                            ..., new_block_idx, :, :, :
                        ] = self.density_plane_all[idx_plane][neighbour].to(self.args.device, non_blocking=True)
                        self.buffers[buffer_idx].app_plane[idx_plane][..., new_block_idx, :, :, :] = self.app_plane_all[
                            idx_plane
                        ][neighbour].to(self.args.device, non_blocking=True)

                    density_handle = broadcast(
                        tensor=self.buffers[buffer_idx].density_plane[idx_plane][:, new_block_idx, :, :, :],
                        parallel_group=ParallelGroup.ProcessesPerNode,
                        async_op=True,
                    )
                    app_handle = broadcast(
                        tensor=self.buffers[buffer_idx].app_plane[idx_plane][:, new_block_idx, :, :, :],
                        parallel_group=ParallelGroup.ProcessesPerNode,
                        async_op=True,
                    )
                    handles.append(density_handle)
                    handles.append(app_handle)
        return handles

    @torch.no_grad()
    def calculate_d2d(self, buffer_idx, neighbours, neighbours_already_have):
        """
        Desc:
            Because some blocks already on device, so only need to copy them from one address to another.

        Args:
            buffer_idx (int): the buffer index within buffers, specific which buffer to be updated.
            neighbours (list[int]): Specify all blocks to be used to update the buffer.
                If block already on device, it will be used in d2d thread, or it will be used in h2d thread.
            neighbours_already_have (list[int]): Specify the block to be used in d2d thread.
        """
        with torch.cuda.stream(self.d2d_stream):
            for new_block_idx, neighbour in enumerate(neighbours):
                if neighbour in neighbours_already_have:
                    old_block_idx = self.buffers[self.current_buffer_idx].neighbours.index(neighbour)
                    for idx_plane in range(len(self.density_plane_all)):
                        self.buffers[buffer_idx].density_plane[idx_plane][:, new_block_idx, :, :, :].copy_(
                            self.buffers[self.current_buffer_idx].density_plane[idx_plane][..., old_block_idx, :, :, :],
                            non_blocking=True,
                        )
                        self.buffers[buffer_idx].app_plane[idx_plane][:, new_block_idx, :, :, :].copy_(
                            self.buffers[self.current_buffer_idx].app_plane[idx_plane][..., old_block_idx, :, :, :],
                            non_blocking=True,
                        )

    def update_plane_buffer2(self, buffer_idx, neighbours, nccl_only=True):
        """
        Desc:
            Using d2d/h2d copy to update the param on device.

        Args:
            buffer_idx (int): the buffer index within buffers, specific which buffer to be updated.
            neighbours (list[int]): Specify all blocks to be used to update the buffer.
                If block already on device, it will be used in d2d thread, or it will be used in h2d thread.
            nccl_only (bool): used to distinguish whether to do the h2d operation. If it's True, only do
                the nccl operations.
        """
        neighbours_to_load, neighbours_already_have = self.calculate_list_diff(
            neighbours, self.buffers[self.current_buffer_idx].neighbours
        )
        d2d_thread = threading.Thread(
            target=self.calculate_d2d,
            args=(
                buffer_idx,
                neighbours,
                neighbours_already_have,
            ),
        )
        d2d_thread.start()
        return self.calculate_h2d(buffer_idx, neighbours, neighbours_to_load, nccl_only)

    def update_line_buffer(self, buffer_idx, neighbours):
        """
        Desc:
            Choose the correct line param to update the buffer.

        Args:
            buffer_idx (int): the buffer index within buffers, specific which buffer to be updated.
            neighbours (list[int]): Specify all blocks to be used to update the buffer.
                for line params, only do the d2d copy.
        """
        for idx_line in range(len(self.density_line_all)):
            self.buffers[buffer_idx].density_line[idx_line] = self.density_line_all[idx_line][..., neighbours]
            self.buffers[buffer_idx].app_line[idx_line] = self.app_line_all[idx_line][..., neighbours]

    def assign_buffer_metadata(self, buffer_idx, neighbours, block_idx):
        """
        Desc:
            Calculate the metadata and assign to buffer. all metadata will be used in threshold trigger
            and param update.

        Args:
            buffer_idx (int): the buffer index within buffers, specific which buffer to be updated.
            neighbours (list[int]): Specify all blocks to be used to update the buffer.
            block_idx (int): the central block index of the buffer's local plane.
        """
        self.buffers[buffer_idx].neighbours = neighbours
        self.buffers[buffer_idx].center_block_idx = block_idx
        self.buffers[buffer_idx].center = self.calculate_block_center(block_idx, device=self.args.device)

    @torch.no_grad()
    def broadcast_full_buffers(self):
        """
        Desc:
            broadcast area from rank 0 to other rank within one node.
        """
        buffer_idx = self.current_buffer_idx

        density_plane = self.buffers[buffer_idx].density_plane
        app_plane = self.buffers[buffer_idx].app_plane

        for idx in range(len(density_plane)):
            broadcast(
                density_plane[idx],
                parallel_group=ParallelGroup.ProcessesPerNode,
                async_op=False,
            )

        for idx in range(len(app_plane)):
            broadcast(
                app_plane[idx],
                parallel_group=ParallelGroup.ProcessesPerNode,
                async_op=False,
            )

    @torch.no_grad()
    def init_buffers(self, pose_o, update_plane=False):
        """
        Desc:
            Initialize the buffers using given pose.

        Args:
            pose_o (torch.Tensor): (x,y,z) of the camera's coordinate.
            update_plane (bool): If True, using correct plane param to update the buffer.
        """
        if self.load_offset is None:
            self.calculate_load_offset()

        self.create_empty_buffers()

        block_idx = self.calculate_block_idx(pose_o)
        neighbours = self.calculate_block_neighbours(block_idx)

        buffer_idx = self.current_buffer_idx
        self.assign_buffer_metadata(buffer_idx, neighbours, block_idx)
        self.update_line_buffer(buffer_idx, neighbours)
        if update_plane:
            self.update_plane_buffer(buffer_idx, neighbours)
        self.broadcast_full_buffers()

        return True

    @torch.no_grad()
    def update_buffers(self, pose_o, nccl_only=True):
        """
        Desc:
            Update the buffers using given pose.

        Args:
            pose_o (torch.Tensor): (x,y,z) of the camera's coordinate.
            nccl_only (bool): used to distinguish whether to do the h2d operation. If it's True, only do
                the nccl operations.
        """
        if self.load_offset is None:
            self.calculate_load_offset()

        center_block_idx = self.buffers[self.current_buffer_idx].center_block_idx
        next_block_idx = self.calculate_next_block_idx(pose_o)
        if torch.equal(center_block_idx, next_block_idx):
            return False, []

        neighbours = self.calculate_block_neighbours(next_block_idx)
        buffer_idx = self.current_buffer_idx ^ 1
        self.assign_buffer_metadata(buffer_idx, neighbours, next_block_idx)
        self.update_line_buffer(buffer_idx, neighbours)

        with self.preload_lock:
            handles = self.update_plane_buffer2(buffer_idx, neighbours, nccl_only=nccl_only)

        return True, handles

    def create_empty_buffers(self):
        """
        Desc:
            use random block to initialize the buffers.

        Note:
            because neighbours is meaningless, the plane and line cannot be used directly.
        """
        for j in [0, 1]:
            buffer_idx = self.current_buffer_idx ^ j

            neighbours = list(range(self.neighbour_size))
            density_plane_blocks, app_plane_blocks = self.create_new_plane_stack(neighbours)
            density_line_blocks, app_line_blocks = self.create_new_line_stack(neighbours)
            self.buffers[buffer_idx].density_plane = density_plane_blocks.to(self.args.device, non_blocking=True)
            self.buffers[buffer_idx].app_plane = app_plane_blocks.to(self.args.device, non_blocking=True)
            self.buffers[buffer_idx].density_line = density_line_blocks
            self.buffers[buffer_idx].app_line = app_line_blocks
        torch.cuda.synchronize()

    def init_svd_volume(self, device):
        """
        Desc:
            Init svd volume parameters.

        Args:
            device (torch.device): Device that the model runs on
        """
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)

        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp) * len(self.resMode), self.app_dim, bias=False).to(device)

        if self.nonlinear_density:
            self.basis_den = torch.nn.Linear(sum(self.density_n_comp) * len(self.resMode), 1, bias=False).to(device)
        if self.args.encode_app:
            self.embedding_app = torch.nn.Embedding(11500, 48).to(device)

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
        if not self.args.ckpt.endswith("-wo_plane.th"):
            ckpt_fp = self.args.ckpt[:-3] + "-wo_plane.th"
        else:
            ckpt_fp = self.args.ckpt
        assert os.path.exists(ckpt_fp), f"{ckpt_fp} not exists !"

        ckpt = torch.load(ckpt_fp, map_location="cpu")
        for i in range(len(self.vecMode)):
            for j in range(len(self.resMode)):
                self.plane_dim = 3 if len(ckpt["gridShape"][f"density_plane.{j}"]) == 5 else 2
                planeSize = ckpt["gridShape"][f"density_plane.{j}"][-self.plane_dim :]
                lineSize = ckpt["gridShape"][f"density_line.{j}"][-2:]
                if should_load_full_plane():
                    plane_coef.append(torch.nn.Parameter(scale * torch.zeros((1, n_component[i], *planeSize))))
                else:
                    # create empty cache for broadcast.
                    plane_coef.append(
                        torch.nn.Parameter(
                            scale * torch.zeros((1, n_component[i], self.neighbour_size, *planeSize[1:]))
                        )
                    )
                line_coef.append(torch.nn.Parameter(scale * torch.zeros((1, n_component[i], *lineSize))))
        del ckpt

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    # optimization
    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
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

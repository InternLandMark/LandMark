# pylint: disable=W0123,E1102
import gc
import os
import time
from abc import abstractmethod
from typing import Callable, Union

import numpy as np
import torch
from psutil import virtual_memory

from app.models.gridnerf_branch_parallel import DistRenderGridNeRFBranchParallel
from app.models.gridnerf_elastic import DistRenderGridNeRFElastic
from app.models.gridnerf_elastic_cuda import DistRenderGridNeRFElasticCuda
from app.models.gridnerf_sequential import GridNeRF
from app.models.gridnerf_tensor_parallel import DistRenderGridNeRFTensorParallel
from dist_render.comm.communication import all_gather, broadcast
from dist_render.comm.parallel_context import ParallelContext, ParallelGroup
from dist_render.comm.profiler import TPCommunicationProfiler
from dist_render.comm.utils import (
    kwargs_tensors_to_device,
    rm_ddp_prefix_in_state_dict_if_present,
)
from dist_render.ddp_infer.editer import Editer


class AbstractDDPInfererImpl:
    """
    Abstract class of model infer operations.
    """

    def __init__(self, context, profile_stages=False) -> None:
        self._chunk_size = context.args_nerf.render_batch_size
        self._args = context.args_nerf
        self._tensor_parallel = False
        self._forward: Union[Callable, None] = None
        self._profile_stages = profile_stages

    @abstractmethod
    def load_model(self, H, W):
        """
        load model for per rank.
        """
        pass

    @property
    def args(self):
        return self._args

    @property
    def tensor_parallel(self):
        return self._tensor_parallel

    def set_pose(self, pose):
        pass

    def edit_model(self, edit_mode):
        pass

    def set_forward_func(self, func):
        self._forward = func

    def remove_old_kwargs(self, kwargs):
        if "shadingMode" in kwargs:
            kwargs.pop("shadingMode")
        return kwargs

    def set_distributed_args(self, data_parallel_local_rank, rank, data_parallel_group_world_size):
        """
        set parallel attron args.

        Args:
            data_parallel_local_rank(int): the local rank in data parallel group.
            rank(int): gobal rank.
            data_parallel_group_world_size(int): the world size of data parallel group.
        """
        if ParallelContext().get_data_parallel_size() >= 1:
            self._args.local_rank = data_parallel_local_rank
            self._args.rank = rank
            self._args.world_size = data_parallel_group_world_size
            self._args.device = "cuda"
            self._args.distributed = True

        self._tensor_parallel = ParallelContext().get_tensor_parallel_size() > 1
        if self._tensor_parallel:
            self._args.tensor_parallel_local_rank = ParallelContext().get_local_rank(
                ParallelGroup.ProcessesInTenParalGroup
            )
            self._args.tensor_parallel_group_world_size = ParallelContext().get_group_world_size(
                ParallelGroup.ProcessesInTenParalGroup
            )
            self._args.tensor_parallel_group = ParallelContext().get_group(ParallelGroup.ProcessesInTenParalGroup)
        # TODO: set p cluster distributed attributes for args

    def renderer_fn(self, rays, **kwargs):
        """
        nerf module forward.

        Args:
            rays(Tensor): rays generated from pose.

        Returns:
            Tensor: rgb.
        """
        all_ret = []
        N_rays_all = rays.shape[0]
        for chunk_idx in range(N_rays_all // self._chunk_size + int(N_rays_all % self._chunk_size > 0)):
            rays_chunk = rays[chunk_idx * self._chunk_size : (chunk_idx + 1) * self._chunk_size]
            with torch.inference_mode():
                ret = self._forward(rays_chunk, **kwargs)

            if isinstance(ret, tuple) and "rgb_map" in ret[0]:
                all_ret.append(ret[0]["rgb_map"])
            elif "rgb_map" in ret:
                all_ret.append(ret["rgb_map"])
            else:
                print(ret, flush=True)
                raise Exception("cannot find correct key for ret")
        all_ret = torch.cat(all_ret, 0)
        return all_ret


class TorchNerfDDPInfererImpl(AbstractDDPInfererImpl):
    """
    Infer operations of non-branchparallel torch model.
    """

    def __init__(self, context) -> None:
        super().__init__(context)

        self._model = None
        self._model_path = context.model_path

    def load_model(self, H, W):
        if self._model is None:
            ckpt = torch.load(self._model_path, map_location=self._args.device)
            kwargs = ckpt["kwargs"]
            kwargs.update({"device": self._args.device, "args": self._args, "is_train": False})
            kwargs = self.remove_old_kwargs(kwargs)
            self._model = GridNeRF(**kwargs)
            self._model.load(ckpt)
            self.set_forward_func(self._model)

    def renderer_fn(self, rays, N_samples=-1, white_bg=True, is_train=False, app_code=None):  # pylint: disable=W0613
        return super().renderer_fn(rays, N_samples=N_samples, white_bg=white_bg, is_train=is_train)


class MultiBlockTorchNerfDDPInfererImpl(AbstractDDPInfererImpl):
    """
    Infer operations of multi-block torch model.
    """

    def __init__(self, context, profile_stages=False) -> None:
        super().__init__(context, profile_stages)

        self._model = None
        self._model_path = context.model_path
        self._enable_edit_mode = context.enable_edit_mode
        if self._enable_edit_mode:
            self._editer = Editer()

    def load_model(self, H, W):
        print("Load model from:", self._model_path)
        ckpt = torch.load(self._model_path, map_location="cpu")
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": self._args.device, "args": self._args, "is_train": False})
        kwargs = self.remove_old_kwargs(kwargs)
        kwargs_tensors_to_device(kwargs, self._args.device)

        self._model = DistRenderGridNeRFBranchParallel(**kwargs)  # pylint: disable=W0123
        rm_ddp_prefix_in_state_dict_if_present(ckpt["state_dict"])
        self._model.load(ckpt)
        self.set_forward_func(self._model)
        print("Load model sucessfully.")

        if self._enable_edit_mode:
            self._editer.load_edit_part(self._model_path)

    def edit_model(self, edit_mode=0):
        if not self._enable_edit_mode:
            return
        self._editer.edit_model(edit_mode, self._model, device=self._args.device)


class MultiBlockTensorParallelTorchNerfDDPInfererImpl(MultiBlockTorchNerfDDPInfererImpl):
    """
    Infer operations of multi-block torch tensor parallel model with encode app.
    """

    def __init__(self, context, profile_stages=False) -> None:
        super().__init__(context, profile_stages)
        self.context = context

    def memory_enough(self):
        # 30
        return virtual_memory()[2] < 30

    def load_model(self, H, W):
        assert H > 0 and W > 0
        part_model_aabb = None
        orderly = False
        if hasattr(self.context, "aabb") and getattr(self.context, "aabb") is not None:
            if not isinstance(self.context.aabb, np.ndarray) and isinstance(self.context.aabb, list):
                part_model_aabb = torch.from_numpy(np.array(self.context.aabb))
            else:
                part_model_aabb = self.context.aabb
            part_model_aabb = part_model_aabb.float().to(self._args.device)
        if hasattr(self.context, "load_orderly"):
            orderly = self.context.load_orderly

        def tensor_parallel_all_gather_func(tensor):
            if self._profile_stages:
                TPCommunicationProfiler.start()
            _, tensor_list = all_gather(tensor=tensor, parallel_group=ParallelGroup.ProcessesInTenParalGroup)
            if self._profile_stages:
                TPCommunicationProfiler.end(tensor=tensor)

            return torch.cat(tensor_list, dim=0)

        def data_parallel2_scatter_func(full_tensor=None):
            data_parallel2_world_size = ParallelContext().get_group_world_size(ParallelGroup.ProcessesInTenParalGroup)
            gap = full_tensor.shape[0] % data_parallel2_world_size
            full_tensor = full_tensor[:-gap] if gap > 0 else full_tensor
            single_length = full_tensor.shape[0] // data_parallel2_world_size
            assert full_tensor.shape[0] % data_parallel2_world_size == 0
            local_rank = ParallelContext().get_local_rank(ParallelGroup.ProcessesInTenParalGroup)
            single_tensor = full_tensor[local_rank * single_length : (local_rank + 1) * single_length]
            return single_tensor, gap

        self._args.tensor_parallel_all_gather_func = tensor_parallel_all_gather_func
        self._args.data_parallel2_scatter_func = data_parallel2_scatter_func

        if ParallelContext().is_group0(ParallelGroup.ProcessesInTenParalGroup):
            if orderly:
                time.sleep(ParallelContext().get_local_rank(ParallelGroup.ProcessesPerNode) * 45)
                while not self.memory_enough():
                    time.sleep(ParallelContext().get_local_rank(ParallelGroup.ProcessesPerNode) * 45)

            print("Load model from:", self._model_path)
            ckpt = torch.load(self._model_path, map_location="cpu")
            kwargs = ckpt["kwargs"]
            kwargs.update({"device": self._args.device, "args": self._args, "is_train": False})
            kwargs = self.remove_old_kwargs(kwargs)
            if part_model_aabb is not None:
                kwargs.update({"aabb": part_model_aabb})

            kwargs_tensors_to_device(kwargs, self._args.device)
            rm_ddp_prefix_in_state_dict_if_present(ckpt["state_dict"])

            split_module_names = ["density_plane", "density_line", "app_plane", "app_line"]
            half_module_names = ["density_plane", "app_plane"]
            keys = sorted(ckpt["state_dict"].keys())
            for key in keys:
                flag = False
                for name in split_module_names:
                    if name in key:
                        flag = True
                if flag:
                    split_tensor = torch.chunk(
                        ckpt["state_dict"][key], self._args.tensor_parallel_group_world_size, dim=1
                    )[self._args.tensor_parallel_local_rank]

                    if self._args.half_precision_param:
                        for name in half_module_names:
                            if name in key:
                                split_tensor = split_tensor.half()
                    ckpt["state_dict"][key] = split_tensor

            self._model = DistRenderGridNeRFTensorParallel(**kwargs)
            self._model.load(ckpt)
            del ckpt
            gc.collect()
        else:
            fp_path = self._model_path[:-3] + "-wo_state_dict.th"
            assert os.path.exists(fp_path)
            print("Load model from:", fp_path)
            ckpt_fp = torch.load(fp_path, map_location="cpu")
            kwargs = ckpt_fp["kwargs"]
            kwargs.update({"device": self._args.device, "args": self._args, "is_train": False})
            kwargs = self.remove_old_kwargs(kwargs)
            if part_model_aabb is not None:
                kwargs.update({"aabb": part_model_aabb})

            kwargs_tensors_to_device(kwargs, self._args.device)
            self._model = DistRenderGridNeRFTensorParallel(**kwargs)
            self._model.update_alpha_mask(ckpt_fp)

        for param in self._model.parameters():
            broadcast(param.data, parallel_group=ParallelGroup.ProcessesSameLocalRankBetTenParalGroup)
        self.set_forward_func(self._model)
        print("Load model sucessfully.")


class KernelFusionNerfDDPInfererImpl(AbstractDDPInfererImpl):
    """
    Infer operations of non-branchparallel kernel fusion model.
    """

    def __init__(self, context) -> None:
        super().__init__(context)
        from dist_render.kernel.cuda_render_extend import (
            tensorf_cuda_init,
            tensorf_part1_cuda,
        )

        self._tensorf = None
        self._density_plane_line_sum_path = context.density_plane_line_sum_path
        self._th_mdoel_path = context.th_model_path
        self.tensorf_cuda_init = tensorf_cuda_init
        self.set_forward_func(tensorf_part1_cuda)

    def set_pose(self, pose):
        self._tensorf.pose = pose

    def load_model(self, H, W):
        # Load tensorf
        ckpt = torch.load(self._th_mdoel_path, map_location=self._args.device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": self._args.device, "args": self._args, "is_train": False})
        kwargs = self.remove_old_kwargs(kwargs)
        self._tensorf = GridNeRF(**kwargs)
        self._tensorf.load(ckpt)

        self._tensorf.fullgrid = 1
        if self._tensorf.fullgrid:
            inputs = np.load(self._density_plane_line_sum_path)
            self._tensorf.density_plane_line_sum = torch.from_numpy(inputs["density_plane_line_sum"]).to(
                self._args.device
            )
            torch.cuda.empty_cache()
            del self._tensorf.density_plane
            del self._tensorf.density_line
            del self._tensorf.alphaMask.alpha_volume

        self._tensorf.eval()
        self.tensorf_cuda_init(self._tensorf, self._chunk_size)
        print("Load ckpt from ", self._th_mdoel_path, flush=True)


class MultiBlockKernelFusionNerfDDPInfererImpl(MultiBlockTorchNerfDDPInfererImpl):
    """
    Infer operations of multi-block kernel fusion model.
    """

    def __init__(self, context) -> None:
        super().__init__(context)
        from dist_render.kernel.cuda_render_extend import (  # noqa: E402  # pylint: disable=C0413
            MultiBlockParallelCuda,
        )

        self._tensorf_cuda = MultiBlockParallelCuda()

    def load_model(self, H, W):
        # Load tensorf
        super().load_model(H, W)
        self._model.eval()

        self._model.fullgrid = 0
        self._model.column_major = 0
        self._tensorf_cuda.allocate_tensors(self._model, self._chunk_size)

        self.set_forward_func(self._tensorf_cuda.forward)
        print("Load model sucessfully.")

    def renderer_fn(self, rays, N_samples=-1, white_bg=True, is_train=False, app_code=None):  # pylint: disable=W0613
        return super().renderer_fn(rays, white_bg=white_bg, app_code=app_code)


class MultiBlockTensorParallelKernelFusionNerfDDPInfererImpl(MultiBlockTensorParallelTorchNerfDDPInfererImpl):
    """
    Infer operations of multi-block tensor parallel kernel fusion model.
    """

    def __init__(self, context) -> None:
        super().__init__(context)

        from dist_render.kernel.cuda_render_extend import (  # noqa: E402  # pylint: disable=C0413
            TPMultiBlockParallelCuda,
        )

        self._tensorf_cuda = TPMultiBlockParallelCuda()

    def load_model(self, H, W):
        super().load_model(H, W)
        self.set_forward_func(self._tensorf_cuda.forward)

        # 0 for v0606
        # 1 for v0607
        self._model.fullgrid = 0
        self._model.column_major = 0
        self._tensorf_cuda.allocate_tensors(self._model, self._chunk_size)

    def renderer_fn(self, rays, N_samples=-1, white_bg=True, is_train=False, app_code=None):  # pylint: disable=W0613
        return super().renderer_fn(rays, white_bg=white_bg, app_code=app_code)


class MovingAreaTorchNerfDDPInfererImpl(AbstractDDPInfererImpl):
    """
    Infer operations of multi-block torch model.
    """

    def __init__(self, context) -> None:
        super().__init__(context)

        self._model = None
        self._model_path = context.model_path

    def load_model(self, H, W):
        local_rank_within_node = ParallelContext().get_local_rank(ParallelGroup.ProcessesPerNode)
        ckpt_fp = self._model_path
        if local_rank_within_node != 0:
            ckpt_fp = self._model_path[:-3] + "-wo_plane.th"
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": "cpu", "args": self._args, "neighbour_size": self._args.neighbour_size})
        kwargs = self.remove_old_kwargs(kwargs)
        kwargs_tensors_to_device(kwargs, self._args.device)

        self._model = DistRenderGridNeRFElastic(**kwargs)  # pylint: disable=W0123

        rm_ddp_prefix_in_state_dict_if_present(ckpt["state_dict"])
        self._model.load(ckpt)
        del ckpt
        gc.collect()
        self._model.permute_and_split_model()
        self._model.update_device(self._args.device)

    def meet_load_threshold(self, pose_o):
        return self._model.meet_load_threshold(pose_o)

    def switch_buffers(self):
        return self._model.switch_buffers()

    def init_buffers(self, pose_o, update_plane=False):
        return self._model.init_buffers(pose_o, update_plane)

    def update_buffers(self, pose_o, nccl_only=False):
        return self._model.update_buffers(pose_o, nccl_only)

    def renderer_fn(
        self,
        rays,
        N_samples=-1,
        white_bg=True,
        is_train=False,
        app_code=0,
    ):  # pylint: disable=W0613
        all_ret = []
        N_rays_all = rays.shape[0]
        for chunk_idx in range(N_rays_all // self._chunk_size + int(N_rays_all % self._chunk_size > 0)):
            rays_chunk = rays[chunk_idx * self._chunk_size : (chunk_idx + 1) * self._chunk_size]
            ret = self._model(rays_chunk)

            if isinstance(ret, tuple) and "rgb_map" in ret[0]:
                all_ret.append(ret[0]["rgb_map"])
            elif "rgb_map" in ret:
                all_ret.append(ret["rgb_map"])
            else:
                print(ret, flush=True)
                raise Exception("cannot find correct key for ret")
        all_ret = torch.cat(all_ret, 0)
        return all_ret


class MovingAreaCudaKernelNerfDDPInfererImpl(AbstractDDPInfererImpl):
    """
    Infer operations of multi-block torch model.
    """

    def __init__(self, context) -> None:
        super().__init__(context)

        self._model = None
        self._model_path = context.model_path

    def load_model(self, H, W):
        local_rank_within_node = ParallelContext().get_local_rank(ParallelGroup.ProcessesPerNode)
        ckpt_fp = self._model_path
        if local_rank_within_node != 0:
            ckpt_fp = self._model_path[:-3] + "-wo_plane.th"
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": "cpu", "args": self._args, "neighbour_size": self._args.neighbour_size})
        kwargs = self.remove_old_kwargs(kwargs)
        kwargs_tensors_to_device(kwargs, self._args.device)

        self._model = DistRenderGridNeRFElasticCuda(**kwargs)  # pylint: disable=W0123

        rm_ddp_prefix_in_state_dict_if_present(ckpt["state_dict"])
        self._model.load(ckpt)
        del ckpt
        gc.collect()
        self._model.permute_and_split_model()
        self._model.update_device(self._args.device)
        self._model.allocate_tensors(self._chunk_size)

    def meet_load_threshold(self, pose_o):
        return self._model.meet_load_threshold(pose_o)

    def switch_buffers(self):
        return self._model.switch_buffers()

    def init_buffers(self, pose_o, update_plane=False):
        return self._model.init_buffers(pose_o, update_plane)

    def update_buffers(self, pose_o, nccl_only=False):
        return self._model.update_buffers(pose_o, nccl_only)

    def renderer_fn(
        self,
        rays,
        N_samples=-1,
        white_bg=True,
        is_train=False,
        app_code=0,
    ):  # pylint: disable=W0613
        all_ret = []
        N_rays_all = rays.shape[0]
        self._model.allocate_buffers(self._chunk_size)
        for chunk_idx in range(N_rays_all // self._chunk_size + int(N_rays_all % self._chunk_size > 0)):
            rays_chunk = rays[chunk_idx * self._chunk_size : (chunk_idx + 1) * self._chunk_size]
            ret = self._model(rays_chunk)
            if isinstance(ret, tuple) and "rgb_map" in ret[0]:
                all_ret.append(ret[0]["rgb_map"])
            elif "rgb_map" in ret:
                all_ret.append(ret["rgb_map"])
            else:
                print(ret, flush=True)
                raise Exception("cannot find correct key for ret")
        all_ret = torch.cat(all_ret, 0)
        return all_ret

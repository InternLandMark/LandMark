import math
import threading

import torch

from dist_render.comm.communication import broadcast, gather, scatter
from dist_render.comm.dynamic_loader import (
    DynamicLoader,
    expand_rays,
    generate_rays_lazy,
)
from dist_render.comm.env import EnvSetting
from dist_render.comm.factory import dataset_factory, inferer_impl_factory
from dist_render.comm.parallel_context import ParallelContext, ParallelGroup
from dist_render.comm.profiler import PipeStagesProfiler, ProfileStageType
from dist_render.comm.singleton import SingletonMeta
from dist_render.comm.types import DatasetType, ModelType
from dist_render.dataset.utils import generate_rays


class AbstractNerfDDPInferer(metaclass=SingletonMeta):
    """
    Abstract class of nerf model inferer.
    """

    def __init__(self, model_type: ModelType, profile_stages=False) -> None:
        self.shape = None
        self.distributed_args_set = False
        self.profile_stages = profile_stages
        self.inferer_impl = inferer_impl_factory(model_type, profile_stages)

        self.comm_stream = torch.cuda.Stream()
        if self.inferer_impl.args.dynamic_fetching:
            self.elastic_loader = DynamicLoader()
            self.first_time_load = True

    def prepare_distribued_and_model(self, H, W, data_parallel_local_rank, rank, data_parallel_group_world_size):
        """
        set dist attr on args and load model for per rank.
        """
        if not self.distributed_args_set:
            self.inferer_impl.set_distributed_args(data_parallel_local_rank, rank, data_parallel_group_world_size)
            self.distributed_args_set = True
        self.inferer_impl.load_model(H, W)

    def get_local_rays_shape(self, H, W):
        """
        get rays shape single data parallel rank.
        """
        if self.shape is None:
            self.shape = [
                math.ceil(W * H / self.inferer_impl.args.world_size),
                3,
            ]
            self.shape = tuple(self.shape)
        return self.shape

    def split_rays(self, rays):
        """
        DESC:
            padding rays to ensure it's divisible by `world_size`, then divide it into list.
        """
        # notice: world_size is **data parallel size**
        world_size = self.inferer_impl.args.world_size
        padding_size = rays.shape[0] % world_size

        if padding_size > 0:
            padding_size = world_size - padding_size
            padding_tensor = torch.zeros((padding_size, 6), device=self.inferer_impl.args.device)
            rays = torch.cat((rays, padding_tensor), dim=0)
        rays_list = list(torch.split(rays, rays.shape[0] // world_size, dim=0))
        return rays_list, padding_size

    def shrink_rays(self, rays):
        rays, app_code, edit_mode, pose_o, focal = (
            rays[:-1, :],
            rays[-1, 0].long(),
            rays[-1, 1],
            rays[-1, 2:5],
            rays[-1, 5],
        )
        return rays, app_code, edit_mode, pose_o, focal

    def evenly_split_rays(self, rays):
        dp_size = ParallelContext().get_data_parallel_size()
        dp_group_id = ParallelContext().get_local_rank(ParallelGroup.ProcessesSameLocalRankBetTenParalGroup)
        sample_per_group = math.ceil(rays.shape[0] / dp_size)
        padding_size = sample_per_group * dp_size - rays.shape[0]
        if padding_size > 0:
            rays = torch.cat(
                (rays, torch.zeros((padding_size, 6), dtype=torch.float32, device=self.inferer_impl.args.device))
            )
        rays = rays.view(sample_per_group, dp_size, 6)
        rays.transpose_(0, 1)
        rays = rays[dp_group_id].clone()
        return rays, padding_size

    def evenly_merge_rgb(self, rgb):
        dp_size = ParallelContext().get_data_parallel_size()
        rgb = rgb.view(dp_size, -1, 3)
        rgb.transpose_(0, 1)
        rgb = rgb.reshape(-1, 3)
        return rgb


class OtherRankNerfDDPInferer(AbstractNerfDDPInferer):
    """
    Other rank inferer.
    """

    def __init__(self, model_type: ModelType) -> None:
        super().__init__(model_type)
        self.all_ret = None

    def allocate_all_ret(self, H, W):
        """
        allocate results tensor just once.
        """
        if self.all_ret is None:
            self.all_ret = [
                torch.zeros(self.get_local_rays_shape(H, W), dtype=torch.float32, device=self.inferer_impl.args.device)
                for _ in range(self.inferer_impl.args.world_size)
            ]
        return self.all_ret

    def render_other_rank(self, H, W, N_samples=-1, white_bg=True):
        """
        worker rank rendering.

        Args:
            H(int): output image height.
            W(int): output image width.
            N_samples(int): The number of samples to take along each ray.
            white_bg(bool): use white as background.
        """
        rays_size = W * H

        if self.inferer_impl.args.dynamic_fetching:
            rays_size = 4
        else:
            rays_size = math.ceil(rays_size / self.inferer_impl.args.world_size) + 1
        rays_with_sigal = torch.zeros((rays_size, 6), dtype=torch.float32, device=self.inferer_impl.args.device)
        gather_handle = None

        i = 0
        while True:
            rays = rays_with_sigal
            if gather_handle is not None:
                gather_handle.wait()
                gather_handle = None

            rays_list = None
            if self.inferer_impl.args.dynamic_fetching:
                broadcast(rays, parallel_group=ParallelGroup.AllProcesses)
            else:
                if ParallelContext().is_in_group(ParallelGroup.ProcessesInDataParalGroup):
                    scatter(
                        tensor=rays,
                        scatter_list=rays_list,
                        async_op=False,
                        parallel_group=ParallelGroup.ProcessesInDataParalGroup,
                    )
                broadcast(tensor=rays, parallel_group=ParallelGroup.ProcessesInTenParalGroup)
            app_code, edit_mode = None, None
            rays, app_code, edit_mode, pose_o, focal = self.shrink_rays(rays)
            if self.inferer_impl.args.dynamic_fetching:
                self.elastic_loader.change_load_status(self.inferer_impl, pose_o)

                pose = rays[:, :5]
                rays = generate_rays(pose, H, W, focal)
                rays, _ = self.evenly_split_rays(rays)

            if self.inferer_impl.args.dynamic_fetching:
                if self.elastic_loader.load_thread is not None and not self.inferer_impl._model.lock_by_render_thread:
                    self.elastic_loader.load_thread.join()

            self.inferer_impl.edit_model(edit_mode.item())
            ret = self.inferer_impl.renderer_fn(
                rays=rays,
                N_samples=N_samples,
                white_bg=white_bg,
                is_train=False,
                app_code=app_code,
            )

            if ParallelContext().is_in_group(ParallelGroup.ProcessesInDataParalGroup):
                gather_handle = gather(
                    tensor=ret, gather_list=None, async_op=True, parallel_group=ParallelGroup.ProcessesInDataParalGroup
                )

            i += 1
            if EnvSetting.CI_TEST_PICTURES > 0 and i >= EnvSetting.CI_TEST_PICTURES:
                break


class MainRankNerfDDPInferer(AbstractNerfDDPInferer):
    """
    Main rank inferer.
    """

    def __init__(self, model_type: ModelType, buffer_len=3, profile_stages=False) -> None:
        super().__init__(model_type, profile_stages)
        self.buffer_len = buffer_len
        self.buffer_idx = 0
        self.buffers = []
        self.buffer_semaphore = threading.Semaphore(value=buffer_len)
        self.gather_handle = None
        self.img_cpu = None

    def allocate_buffers(self, H, W):
        """
        get buffer tensors and allocate just once.
        """
        self.buffer_semaphore.acquire()  # pylint: disable=R1732
        if len(self.buffers) == 0:
            for _ in range(self.buffer_len):
                all_ret = [
                    torch.zeros(
                        self.get_local_rays_shape(H, W), dtype=torch.float32, device=self.inferer_impl.args.device
                    )
                    for _ in range(self.inferer_impl.args.world_size)
                ]
                self.buffers.append(all_ret)
        res = self.buffers[self.buffer_idx]
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_len
        return res

    def release_buffers(self):
        """
        release semaphore of buffer tensors.
        """
        self.buffer_semaphore.release()

    def model_infer_preprocess(self, pose, H, W, focal=None):
        """
        preprocess before module forward.

        Args:
            pose(list): pose info read from dataset or get from user.
            H(int): output image height.
            W(int): output image width.
            focal(float): focal value.
        """
        if self.profile_stages:
            if PipeStagesProfiler.module_profiler is None:
                PipeStagesProfiler.module_profiler = self.inferer_impl.module_profiler
            PipeStagesProfiler.start(ProfileStageType.Preprocess)
        self.inferer_impl.set_pose(pose)

        pose = torch.from_numpy(pose).type(torch.float32).cuda()
        rays_list, padding_size = [], 0
        if self.inferer_impl.args.dynamic_fetching:
            # delay pose2rays to reduce comm.
            rays_list, padding_size = generate_rays_lazy(pose)
        else:
            rays = generate_rays(pose, H, W, focal)  # ray is on device
            rays_list, padding_size = self.split_rays(rays)

        expand_rays(rays_list, pose, self.inferer_impl.args.device)

        all_ret = self.allocate_buffers(H, W)

        if self.profile_stages:
            PipeStagesProfiler.end(ProfileStageType.Preprocess)
        return [rays_list, padding_size, all_ret, torch.cuda.current_stream()]

    def model_infer(self, data, H, W, N_samples=-1, white_bg=True):
        """
        module forward.

        Args:
            data(tuple): preprocessed data.
            H(int): output image height.
            W(int): output image width.
            N_samples(int): The number of samples to take along each ray.
            white_bg(bool): use white as background.
        """
        rays_list, padding_size, all_ret, pre_stream = (
            data[0],
            data[1],
            data[2],
            data[3],
        )

        torch.cuda.current_stream().wait_stream(pre_stream)
        pre_stream.synchronize()

        if self.profile_stages:
            PipeStagesProfiler.start(ProfileStageType.ModelInfer)

        if self.gather_handle is not None:
            self.gather_handle.wait()

        # rays=rays_list[0], MainRank saves memory by reusing
        rays = rays_list[0]
        if self.inferer_impl.args.dynamic_fetching:
            broadcast(rays, parallel_group=ParallelGroup.AllProcesses)
        else:
            assert ParallelContext().is_in_group(ParallelGroup.ProcessesInDataParalGroup)
            scatter(
                tensor=rays,
                scatter_list=rays_list,
                async_op=False,
                parallel_group=ParallelGroup.ProcessesInDataParalGroup,
            )
            assert ParallelContext().is_group_rank0(
                ParallelGroup.ProcessesInTenParalGroup
            )  # !important, parallel context doesn't gaurantee this condition
            broadcast(tensor=rays, parallel_group=ParallelGroup.ProcessesInTenParalGroup)

        rays, app_code, edit_mode, pose_o, focal = self.shrink_rays(rays)
        if self.inferer_impl.args.dynamic_fetching:

            self.elastic_loader.change_load_status(self.inferer_impl, pose_o)
            pose = rays[:, :5]
            rays = generate_rays(pose, H, W, focal)
            rays, padding_size = self.evenly_split_rays(rays)

        if self.inferer_impl.args.dynamic_fetching:
            if self.elastic_loader.load_thread is not None and not self.inferer_impl._model.lock_by_render_thread:
                self.elastic_loader.load_thread.join()

        # forward
        if self.profile_stages:
            PipeStagesProfiler.start(ProfileStageType.Render)
        self.inferer_impl.edit_model(edit_mode)
        ret = self.inferer_impl.renderer_fn(
            rays=rays,
            N_samples=N_samples,
            white_bg=white_bg,
            is_train=False,
            app_code=app_code,
        )
        if self.profile_stages:
            PipeStagesProfiler.end(ProfileStageType.Render)

        self.gather_handle = gather(
            gather_list=all_ret, tensor=ret, async_op=True, parallel_group=ParallelGroup.ProcessesInDataParalGroup
        )
        if self.profile_stages:
            PipeStagesProfiler.end(ProfileStageType.ModelInfer)

        return [all_ret, padding_size, self.gather_handle]

    def model_infer_postprocess(self, ret, H, W):
        """
        postprocess after module forward.

        Args:
            ret(tuple): module forward output.
            H(int): output image height.
            W(int): output image width.
        """
        all_ret, padding_size, gather_handle = ret[0], ret[1], ret[2]
        if self.profile_stages:
            PipeStagesProfiler.start(ProfileStageType.Postprocess)
        if gather_handle is not None:
            gather_handle.wait()  # async pipe

        torch.cuda.current_stream().synchronize()
        result = torch.cat(all_ret, 0)
        if self.inferer_impl.args.dynamic_fetching:
            result = self.evenly_merge_rgb(result)

        self.release_buffers()
        if padding_size > 0:
            result = result[:-padding_size, :]
        result = result.reshape(H, W, 3) * 255
        result = torch.cat([result, torch.ones((H, W, 1), device=self.inferer_impl.args.device) * 255], dim=-1)
        result = result.byte().detach()
        if self.img_cpu is None:
            self.img_cpu = torch.ones_like(result, device="cpu").pin_memory()
        self.img_cpu = result.to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()
        result = self.img_cpu.numpy().copy()
        if self.profile_stages:
            PipeStagesProfiler.end(ProfileStageType.Postprocess)
        return result


def generate_hw(dataset_type):
    """
    return image height and width.
    """
    if EnvSetting.RENDER_1080P:
        assert dataset_type is None or dataset_type in DatasetType
        return 1080, 1920
    else:
        dataset_trans_inst = dataset_factory(dataset_type)
        return dataset_trans_inst.img_hw


@torch.no_grad()
def model_infer_preprocess(
    single_pose, H, W, model_type, data_parallel_local_rank, rank, data_parallel_group_world_size, buffer_len=3
):
    """
    wraper api of model preprocess func.
    """
    inferer = MainRankNerfDDPInferer(model_type, profile_stages=EnvSetting.PROFILE_STAGES, buffer_len=buffer_len)
    if not inferer.distributed_args_set:
        inferer.prepare_distribued_and_model(H, W, data_parallel_local_rank, rank, data_parallel_group_world_size)

    focal = 1483.6378 if EnvSetting.RENDER_1080P else single_pose[-1, -1]
    ret = inferer.model_infer_preprocess(single_pose, H, W, focal)
    return ret


@torch.no_grad()
def infer(rays, H, W, model_type):
    """
    wraper api of model forward func on main/master rank.
    """
    inferer = MainRankNerfDDPInferer(model_type, profile_stages=EnvSetting.PROFILE_STAGES)
    ret = inferer.model_infer(rays, H, W)
    return ret


@torch.no_grad()
def model_infer_postprocess(ret, H, W, model_type):
    """
    wraper api of model postprocess func.
    """
    inferer = MainRankNerfDDPInferer(model_type, profile_stages=EnvSetting.PROFILE_STAGES)
    ret = inferer.model_infer_postprocess(ret, H, W)
    return ret


@torch.no_grad()
def render_other_rank(H, W, model_type, data_parallel_local_rank, rank, data_parallel_group_world_size):
    """
    wraper api of model forward func in other/worker rank.
    """
    inferer = OtherRankNerfDDPInferer(model_type)
    if not inferer.distributed_args_set:
        inferer.prepare_distribued_and_model(H, W, data_parallel_local_rank, rank, data_parallel_group_world_size)

    inferer.render_other_rank(H, W)

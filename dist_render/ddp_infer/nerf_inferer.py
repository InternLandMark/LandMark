import math
import threading

import torch
from comm.communication import all_gather, scatter
from comm.env import EnvSetting
from comm.parallel_context import ParallelGroup
from comm.profiler import PipeStagesProfiler, ProfileStageType
from comm.singleton import SingletonMeta
from comm.types import DatasetType, ModelType, dataset_factory, inferer_impl_factory
from dataset.utils import generate_rays


class GlobalArgsManager(metaclass=SingletonMeta):
    """
    Global args manager to change model args when engine is running.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.global_args = {"app_code": 2930, "edit_mode": 0}  # [edit_mode] 0: resetBuild, 1: newBuild, 2: removeBuild

    def get_arg(self, name):
        """
        get nerf changeable args.

        Args:
            name(str): args name.

        Returns:
            int: args value.
        """
        with self.lock:
            return self.global_args[name]

    def set_arg(self, name, value):
        """
        set nerf changeable args value.

        Args:
            name(str): args name.
            value(int): args value.
        """
        with self.lock:
            self.global_args[name] = value


class AbstractNerfDDPInferer(metaclass=SingletonMeta):
    """
    Abstract class of nerf model inferer.
    """

    def __init__(self, model_type: ModelType, profile_stages=False) -> None:
        self.shape = None
        self.distributed_args_set = False
        self.profile_stages = profile_stages
        self.inferer_impl = inferer_impl_factory(model_type, profile_stages)

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
            self.shape = (
                math.ceil(W * H / self.inferer_impl.args.world_size),
                3,
            )
        return self.shape


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
        rays_size = math.ceil(rays_size / self.inferer_impl.args.world_size)
        rays_with_sigal = torch.zeros((rays_size + 1, 6), dtype=torch.float32, device=self.inferer_impl.args.device)

        while True:
            all_ret = self.allocate_all_ret(H, W)
            rays_list = None
            scatter(tensor=rays_with_sigal, scatter_list=rays_list, parallel_group=ParallelGroup.AllProcesses)
            rays = rays_with_sigal[:-1]
            app_code = rays_with_sigal[-1][0].long()
            edit_mode = rays_with_sigal[-1][1]
            self.inferer_impl.edit_model(edit_mode.item())
            ret = self.inferer_impl.renderer_fn(
                rays=rays,
                N_samples=N_samples,
                white_bg=white_bg,
                is_train=False,
                app_code=app_code,
            )

            all_gather(
                tensor_list=all_ret, tensor=ret, async_op=True, parallel_group=ParallelGroup.ProcessesInDataParalGroup
            )


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
        self.last_ret, self.last_gap, self.last_tid = None, None, None
        self.is_first = True

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
        rays = generate_rays(pose, H, W, focal)  # ray is on device
        assert self.inferer_impl.args.distributed
        world_size = self.inferer_impl.args.world_size
        gap_size = rays.shape[0] % world_size

        if gap_size > 0:
            gap_size = world_size - gap_size
            gap_tensor = torch.zeros((gap_size, 6), device=self.inferer_impl.args.device)
            rays = torch.cat((rays, gap_tensor), dim=0)

        rays_list_temp = list(torch.split(rays, rays.shape[0] // world_size, dim=0))

        # Cat sigal
        app_code = GlobalArgsManager().get_arg("app_code")
        edit_mode = GlobalArgsManager().get_arg("edit_mode")
        sigal = torch.cuda.FloatTensor([[app_code, edit_mode, 0, 0, 0, 0]], device=self.inferer_impl.args.device)
        rays_list_sigal = [torch.cat((per_dp_rays, sigal), dim=0) for per_dp_rays in rays_list_temp]

        if self.inferer_impl.tensor_parallel:
            rays_list = []
            for per_dp_rays in rays_list_sigal:
                rays_list.extend([per_dp_rays] * self.inferer_impl.args.tensor_parallel_group_world_size)
        else:
            rays_list = rays_list_sigal

        rays = rays_list_temp[0]
        all_ret = self.allocate_buffers(H, W)

        if self.profile_stages:
            PipeStagesProfiler.end(ProfileStageType.Preprocess)
        app_code = torch.cuda.LongTensor([app_code], device=self.inferer_impl.args.device)
        return [rays, rays_list, gap_size, all_ret, torch.cuda.current_stream(), app_code, edit_mode]

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
        rays, rays_list, gap, all_ret, pre_stream, app_code, edit_mode = (
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
        )

        torch.cuda.current_stream().wait_stream(pre_stream)

        if self.profile_stages:
            PipeStagesProfiler.start(ProfileStageType.ModelInfer)

        # To consider tensor parallel and data parallel, we need to scatter on all processes.
        scatter(
            tensor=rays_list[0],
            scatter_list=rays_list,
            async_op=True,
            parallel_group=ParallelGroup.AllProcesses,
        )

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

        assert ret.shape == self.get_local_rays_shape(H, W), (
            f"W = {W}, H = {H}, slade size = {math.ceil(W*H/self.inferer_impl.args.world_size)}*3, "
            f"but ret.shape = {ret.shape}"
        )

        gather_handle = all_gather(
            tensor_list=all_ret, tensor=ret, async_op=True, parallel_group=ParallelGroup.ProcessesInDataParalGroup
        )
        if self.profile_stages:
            PipeStagesProfiler.end(ProfileStageType.ModelInfer)

        return [all_ret, gap, gather_handle]

    def model_infer_postprocess(self, ret, H, W):
        """
        postprocess after module forward.

        Args:
            ret(tuple): module forward output.
            H(int): output image height.
            W(int): output image width.
        """
        all_ret, gap_size, gather_handle = ret[0], ret[1], ret[2]
        if self.profile_stages:
            PipeStagesProfiler.start(ProfileStageType.Postprocess)
        gather_handle.wait()  # async pipe

        result = torch.cat(all_ret, 0)
        self.release_buffers()
        if gap_size > 0:
            result = result[:-gap_size, :]
        result = result.reshape(H, W, 3) * 255
        result = torch.cat([result, torch.ones((H, W, 1), device=self.inferer_impl.args.device) * 255], dim=-1)
        result = result.byte().detach()
        result = result.to("cpu")  # TODO: tensor.cpu() need to be replaced with customed op.
        event = torch.cuda.Event(blocking=True)
        event.record()
        event.wait()
        result = result.numpy()
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
    return inferer.model_infer_preprocess(single_pose, H, W, focal)


def infer(rays, H, W, model_type):
    """
    wraper api of model forward func on main/master rank.
    """
    inferer = MainRankNerfDDPInferer(model_type, profile_stages=EnvSetting.PROFILE_STAGES)
    return inferer.model_infer(rays, H, W)


def model_infer_postprocess(ret, H, W, model_type):
    """
    wraper api of model postprocess func.
    """
    inferer = MainRankNerfDDPInferer(model_type, profile_stages=EnvSetting.PROFILE_STAGES)
    return inferer.model_infer_postprocess(ret, H, W)


def render_other_rank(H, W, model_type, data_parallel_local_rank, rank, data_parallel_group_world_size):
    """
    wraper api of model forward func in other/worker rank.
    """
    inferer = OtherRankNerfDDPInferer(model_type)
    if not inferer.distributed_args_set:
        inferer.prepare_distribued_and_model(H, W, data_parallel_local_rank, rank, data_parallel_group_world_size)

    inferer.render_other_rank(H, W)

import threading

import torch

from dist_render.comm.global_args import GlobalArgsManager
from dist_render.comm.parallel_context import ParallelContext, ParallelGroup
from dist_render.comm.types import LoadStatus


def expand_rays(rays_list, pose, device):
    """
    Desc:
        Expand the `rays` with signal to control the illumination and so on.

    Args:
        pose (torch.Tensor): The pose indicate the camera's coordinate and orientation.
    """
    pose_o = pose[:, -2]
    focal = pose[-1, -1]

    # add one more ray to control the model loading in all process.
    signal = torch.cuda.FloatTensor([[0, 0, 0, 0, 0, 0]], device=device)

    # Cat sigal : [app_code, edit_mode, pose_o[0], pose_o[1], pose_o[2], focal]
    app_code = GlobalArgsManager().get_arg("app_code")
    edit_mode = GlobalArgsManager().get_arg("edit_mode")
    signal[0][0] = app_code
    signal[0][1] = edit_mode
    signal[0, 2:5] = pose_o
    signal[0][5] = focal

    for i, rays in enumerate(rays_list):
        rays_list[i] = torch.cat((rays, signal), dim=0)


def generate_rays_lazy(pose):
    """
    Desc:
        Broadcast pose across all rank to reduce traffic.

    Args:
        pose (torch.Tensor): The pose indicate the camera's coordinate and orientation.
    """
    rays = torch.zeros((3, 6), dtype=torch.float32, device="cuda")
    rays[:, :5] = pose
    rays_list, gap_size = [rays], 0
    return rays_list, gap_size


def should_load_full_plane():
    ret = False
    if ParallelContext().get_tensor_parallel_size() > 1:
        local_rank_within_tp_group = ParallelContext().get_local_rank(ParallelGroup.ProcessesInTenParalGroup)
        local_rank_within_same_tp_rank_group = ParallelContext().get_local_rank(
            ParallelGroup.ProcessesSameLocalRankBetTenParalGroup
        )
        if local_rank_within_tp_group == local_rank_within_same_tp_rank_group:
            ret = True
    else:
        local_rank_within_node = ParallelContext().get_local_rank(ParallelGroup.ProcessesPerNode)
        if local_rank_within_node == 0:
            ret = True
    return ret


class DynamicLoader:
    """
    Desc:
        Simple state machine to synchronize the rendering thread with loading thread.
    """

    def __init__(self):
        self.load_status = LoadStatus.Init
        self.load_stream = torch.cuda.Stream()
        self.load_thread = None

    def meet_load_threshold(self, inferer_impl, pose_o):
        """
        DESC:
            Check whether we should start load area.
        """
        return inferer_impl.meet_load_threshold(pose_o)

    def switch_buffers(self, inferer_impl):
        """
        DESC:
            execute after load plane and broadcast plane done, then change the current plane to new plane.
        """
        inferer_impl.switch_buffers()

    def change_load_status(self, inferer_impl, pose_o):
        """
        Desc:
            core function to control the loading status, which was used to
            synchronize loading thread and rendering thread.

        Args:
            inferer_impl (AbstractDDPInfererImpl): instance of inferer.
            pose_o (torch.Tensor): (x,y,z) of the camera's coordinate.
        """
        if self.load_status is LoadStatus.Init:
            # load plane to gpu and broadcast to other rank within node.
            self.init_buffers(inferer_impl, pose_o, should_load_full_plane())

            torch.distributed.barrier(group=ParallelContext().get_group(parallel_group=ParallelGroup.AllProcesses))
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.load_status = LoadStatus.IDLE

        if self.load_status is LoadStatus.IDLE:
            if self.meet_load_threshold(inferer_impl, pose_o):
                self.load_status = LoadStatus.Loading
                if not inferer_impl._model.lock_by_render_thread:
                    inferer_impl._model.preload_lock.acquire()
                    inferer_impl._model.lock_by_render_thread = True
                self.load_thread = threading.Thread(
                    target=self.update_buffers,
                    args=(
                        inferer_impl,
                        pose_o,
                        not should_load_full_plane(),
                    ),
                )
                self.load_thread.start()
        elif self.load_status is LoadStatus.Loading:
            if inferer_impl._model.lock_by_render_thread:
                inferer_impl._model.lock_by_render_thread = False
                inferer_impl._model.preload_lock.release()
        elif self.load_status == LoadStatus.LoadDone:
            self.switch_buffers(inferer_impl)
            self.load_status = LoadStatus.IDLE
        else:
            raise Exception(f"not this type of load_status: {self.load_status}")

    @torch.no_grad()
    def wait_handles(self, handles, inferer_impl):
        """
        Desc:
            Waiting for broadcast kernel done.

        Args:
            handles (List[torch.Work]): handle for broadcast function.
            inferer_impl (AbstractDDPInfererImpl): instance of inferer.
        """
        with torch.cuda.stream(self.load_stream):
            for handle in handles:
                handle.wait()
            self.load_status = LoadStatus.LoadDone
        print(f"rank = {inferer_impl.args.rank}, after load, change status to LoadDone", flush=True)

    @torch.no_grad()
    def init_buffers(self, inferer_impl, pose_o, update_plane=False):
        with torch.cuda.stream(self.load_stream):
            inferer_impl.init_buffers(pose_o, update_plane)

    @torch.no_grad()
    def update_buffers(self, inferer_impl, pose_o, nccl_only=True):
        with torch.cuda.stream(self.load_stream):
            should_load, handles = inferer_impl.update_buffers(pose_o, nccl_only)
        if should_load:
            wait_thread = threading.Thread(
                target=self.wait_handles,
                args=(
                    handles,
                    inferer_impl,
                ),
            )
            wait_thread.start()
        else:
            print(f"rank = {inferer_impl.args.rank}, after load, change status to IDLE", flush=True)
            self.load_status = LoadStatus.IDLE

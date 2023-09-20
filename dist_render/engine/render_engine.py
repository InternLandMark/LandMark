import os
import time
from abc import abstractmethod

from comm.parallel_context import ParallelContext, ParallelGroup, init_parallel_context
from comm.types import DatasetType, ModelType, dataset_factory, runner_factory
from ddp_infer.context import NerfContext
from ddp_infer.nerf_inferer import generate_hw, render_other_rank


class AbstractRenderEngine:
    """
    Abstract engine class.
    """

    def __init__(self, dataset: DatasetType, model_type: ModelType, use_multistream=False) -> None:
        self.dataset_type = dataset
        self.model_type = model_type

        self.use_multistream = use_multistream

    @abstractmethod
    def start(self, test=False):
        pass


class BaseMainRankRenderEngine(AbstractRenderEngine):
    """
    Base main rank engine class to create and execute runner.
    It includes some testing operations.
    """

    def __init__(
        self,
        dataset: DatasetType,
        model_type: ModelType,
        use_multistream=False,
        save_png=False,
        pose_num=-1,
    ) -> None:
        super().__init__(dataset, model_type, use_multistream)
        self.runner = None

        # For test
        self.poses = None
        self.N = None
        self.pose_num = pose_num
        self.pause_running = False
        self.save_png = save_png
        self.save_path = os.path.expanduser(NerfContext.png_saving_path)
        if self.save_png and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def create_runer(self, runner_type, *args, **kwargs):
        """
        create runner according to runner type.
        """
        return runner_factory(runner_type)(*args, **kwargs)

    def prepare_poses(self):
        """
        read poses from city dataset.
        """
        self.dataset_inst = dataset_factory(self.dataset_type)
        self.poses = self.dataset_inst.poses
        self.N = len(self.poses)

    def pose_filter(self, n_interval):
        """
        filter pose by interval and pose num
        """
        test_idxs = list(range(0, self.N, n_interval))
        poses = [self.poses[test_idx] for test_idx in test_idxs]
        if self.pose_num > 0 and len(poses) >= self.pose_num:
            poses = poses[: self.pose_num]
        return poses

    def pause(self):
        # `test` should be true in engine start function
        single_pose = self.poses[0]
        while True:
            if not self.pause_running:
                break
            self.runner.run_batch(single_pose)
            time.sleep(1)

    def is_pausing(self):
        return self.pause_running is True

    def cancel_pause(self):
        self.pause_running = False


class OtherRankRenderEngine(AbstractRenderEngine):
    """Other rank engine."""

    def __init__(self, dataset, model_type, use_multistream=False) -> None:
        super().__init__(dataset, model_type, use_multistream)
        self.H, self.W = generate_hw(self.dataset_type)

    def start(self, test=False):
        """
        start rendering of other/worker rank.
        """
        init_parallel_context()
        render_other_rank(
            self.H,
            self.W,
            self.model_type,
            data_parallel_local_rank=ParallelContext().get_local_rank(ParallelGroup.ProcessesInDataParalGroup),
            rank=ParallelContext().get_rank(),
            data_parallel_group_world_size=ParallelContext().get_group_world_size(
                ParallelGroup.ProcessesInDataParalGroup
            ),
        )

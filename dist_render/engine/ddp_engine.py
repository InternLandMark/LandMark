import os
import time

from comm.types import DatasetType, ModelType, RunnerType
from engine.render_engine import BaseMainRankRenderEngine
from PIL import Image as im


class MainRankDDPRenderEngine(BaseMainRankRenderEngine):
    """
    DDP render engine, create and execute ddp runner.
    """

    def __init__(self, dataset: DatasetType, model_type: ModelType, save_png=False, pose_num=-1) -> None:
        super().__init__(dataset, model_type, save_png=save_png, pose_num=pose_num)

    def start(self, test=False):
        """
        start main/master rank engine runner.

        Args:
            test(bool): whether to run test dataset.
        """
        self.runner = self.create_runer(RunnerType.DDPRunner, self.dataset_type, self.model_type)

        if test:
            self.prepare_poses()
            self.run_test()

    def warm_up(self, pose_num=60):
        """
        warm up iterations.
        """
        # Start the pipeline with model creation.
        for pose in self.poses[:pose_num]:
            _ = self.runner.run_batch(pose)

    def run_for_latency(self):
        """
        test latency.
        """
        latency = []
        # TODO: support profiling pytorch timeline for pure ddp
        # if `EnvSetting.PROFILE_TIMELINE` is True.
        for i, single_pose in enumerate(self.poses):
            last_time = time.time()
            nerf_out = self.runner.run_batch(single_pose)
            latency.append(time.time() - last_time)
            if self.save_png:
                im_png = im.fromarray(nerf_out)
                cur_png_path = os.path.join(self.save_path, f"{i}.png")
                im_png.save(cur_png_path)
                print(cur_png_path, " saved.", flush=True)

        if len(latency) > 0:
            avg_latency = sum(latency) / len(latency)
            print("Latency:", avg_latency, flush=True)
            print("Throughput:", 1 / avg_latency, flush=True)

    def run_test(self):
        """
        test latency using test city dataset
        """
        self.poses = self.pose_filter(n_interval=1)
        print("Pose Num = ", len(self.poses))
        self.warm_up()

        self.run_for_latency()

    def submit(self, pose):
        """
        input a pose and return a rendered img of this pose
        """
        return self.runner.run_batch(pose)

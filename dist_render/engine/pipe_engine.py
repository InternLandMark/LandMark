import os
import time
from queue import Empty
from threading import Thread

from comm.env import EnvSetting
from comm.types import RunnerType
from engine.render_engine import BaseMainRankRenderEngine
from PIL import Image as im


class MainRankPipeRenderEngine(BaseMainRankRenderEngine):
    """
    Pipeline render engine, create and execute multi-thread pipeline runner to overlap preprocess and postprocess.
    """

    def __init__(self, dataset, model_type, save_png=False, async_=False, use_multistream=False, pose_num=-1) -> None:
        super().__init__(dataset, model_type, use_multistream=use_multistream, save_png=save_png, pose_num=pose_num)
        self.running = False
        self.async_ = async_
        self.runner = None
        self.completion_thread = None
        self.completion_map = {}

    def start(self, test=False):
        """
        start main/master rank engine runner.

        Args:
            test(bool): whether to run test dataset.
        """
        self.runner = self.create_runer(
            RunnerType.EasyPipeRunner,
            self.dataset_type,
            self.model_type,
            use_multistream=self.use_multistream,
            profile_timeline=EnvSetting.PROFILE_TIMELINE,
        )
        if self.async_:
            self._start_completion_loop()

        if test:
            self.prepare_poses()
            self.run_test()

    def submit(self, single_pose):
        """
        input a pose and return a uid of this pose.
        """
        return self.runner.run_batch(single_pose)

    def get_result(self):
        """
        get image result from pipeline result queue.
        """
        return self.runner.get_result()

    def warm_up(self, pose_num=60):
        """
        start the pipeline with model creation.
        """
        for pose in self.poses[:pose_num]:
            uid = self.submit(pose)
            if self.async_:
                _ = self.wait(uid)
            else:
                self.runner.get_result()

    def stop(self):
        """
        stop engine runner.
        """
        self.running = False
        self.runner.stop()
        if self.async_:
            self.runner.wait_stop()
            if self.completion_thread is not None:
                self.completion_thread.join()
        print("Stop sucessfully")

    def run_for_latency(self):
        """
        test latency using city dataset.
        """
        latency = []
        for single_pose in self.poses:
            last_time = time.time()
            uid = self.submit(single_pose)
            if self.async_:
                _ = self.wait(uid)
            else:
                _ = self.runner.get_result()
            latency.append(time.time() - last_time)

        if len(latency) > 0:
            print("Latency:", sum(latency) / len(latency), flush=True)

    def run_for_throughput(self):
        """
        test throughput using city dataset.
        """
        start_time = time.time()
        uids = []
        for single_pose in self.poses:
            uid = self.submit(single_pose)
            uids.append(uid)

        latency = []

        loop_num = len(self.poses)
        for i in range(loop_num):
            latency.append(time.time())
            if self.async_:
                nerf_out = self.wait(uids[i])
            else:
                nerf_out = self.runner.get_result()
            latency[i] = time.time() - latency[i]
            if self.save_png:
                im_png = im.fromarray(nerf_out)
                cur_png_path = os.path.join(self.save_path, f"{i}.png")
                im_png.save(cur_png_path)
                print(cur_png_path, " saved.", flush=True)

        total_time = time.time() - start_time
        print("Total time = ", total_time, flush=True)
        print("Throughput = ", len(self.poses) / total_time, flush=True)
        print("Latency with throughput = ", sum(latency) / len(latency), flush=True)

    def run_test(self):
        """
        test latency and throughput using city dataset.
        """
        self.poses = self.pose_filter(n_interval=1)
        print("Pose Num = ", len(self.poses))

        # warm up
        self.warm_up()

        self.run_for_throughput()
        self.run_for_latency()

        self.stop()

    def _completion_loop(self):
        """
        get result image from pipeline runner with a new thread.
        """
        while self.running:
            try:
                result_task = self.runner.get_result_task(block=False)
            except Empty:
                time.sleep(0.001)
                continue

            self.completion_map[result_task.uid] = result_task.value
            time.sleep(0.001)

    def _start_completion_loop(self):
        """
        starting a new thread to get result image from pipeline runner asynchronously.
        """
        self.running = True
        self.completion_thread = Thread(target=self._completion_loop)
        self.completion_thread.start()

    def wait(self, uid):
        """
        get output asynchronously.

        Args:
            uid(str): returned when you submit a pose to pipeline runner.
        """
        assert self.completion_thread.is_alive()
        while True:
            if uid in self.completion_map:
                output = self.completion_map[uid]
                del self.completion_map[uid]
                return output
            time.sleep(0.001)

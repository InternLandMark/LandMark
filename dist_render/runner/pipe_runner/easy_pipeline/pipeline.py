# -*- coding: utf-8 -*-

import multiprocessing as mp
from queue import SimpleQueue

from .worker import ProfileWorkerThread, WorkerThread


class PipelineItem(object):
    """
    Item of pipeline stage.
    """

    def __init__(
        self,
        work_fn,
        init_fn,
        worker_num,
        result_max_length=-1,
        use_multistream=False,
        profile_timeline=False,
    ):
        self.work_fn = work_fn
        self.init_fn = init_fn
        self.worker_num = worker_num
        self.result_max_length = result_max_length
        self.use_multistream = use_multistream
        self.profile_timeline = profile_timeline


class SimplePipeline(object):
    """
    Model infer multi-thread pipeline class.
    It is responsible for creating and starting different pipeline stages.
    """

    def __init__(self, items, job_queue):
        super().__init__()
        self.job_queue = job_queue
        self.thread_pool = []
        self.result_queue = None
        for idx in range(len(items)):
            pipeline_item = items[idx]
            task_thread_pool = []

            # curr and next worker_num will be used to stop processes safely
            curr_worker_num = mp.Value("i", pipeline_item.worker_num)
            if idx < len(items) - 1:
                next_worker_num = mp.Value("i", items[idx + 1].worker_num)
            else:
                next_worker_num = mp.Value("i", 1)

            self.result_queue = SimpleQueue()

            for _ in range(pipeline_item.worker_num):
                # set the `profile_idx` to the worker you want to profile.
                worker_thread_cls = ProfileWorkerThread if pipeline_item.profile_timeline else WorkerThread
                this_thread = worker_thread_cls(
                    pipeline_item.work_fn,
                    pipeline_item.init_fn,
                    self.job_queue,
                    self.result_queue,
                    curr_worker_num,
                    next_worker_num,
                    pipeline_item.use_multistream,
                )

                task_thread_pool.append(this_thread)
            self.job_queue = self.result_queue
            self.thread_pool.append(task_thread_pool)

    def start(self):
        """
        start all stage process threads.
        """
        for task_thread_pool in self.thread_pool:
            for task_thread in task_thread_pool:
                task_thread.run()

    def stop(self):
        """
        stop all threads in pool.
        """
        if len(self.thread_pool) > 0:
            task_thread_pool = self.thread_pool[0]
            for task_thread in task_thread_pool:
                task_thread.put_stop_task()

    def wait_stop(self):
        """
        wait for all threads to stop.
        """
        if len(self.thread_pool) > 0:
            task_thread_pool = self.thread_pool[0]
            for task_thread in task_thread_pool:
                task_thread.wait_stop()

    def get_result_queue(self):
        return self.result_queue

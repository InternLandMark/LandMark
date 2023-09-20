# -*- coding: utf-8 -*-
import os
from threading import Thread

import torch

from .task import StopTask, Task


class Worker(object):
    """
    base pipeline stage worker defination.
    """

    def __init__(self):
        pass

    def process(self, task):
        pass


class SimpleWorker(Worker):
    """
    Executor of different pipeline stage func.
    """

    def __init__(self, work_fn, init_fn=None, use_multistream=False):
        super().__init__()
        self.work_fn = work_fn
        self.resource = None
        self.use_multistream = use_multistream
        if self.use_multistream:
            self.stream = torch.cuda.Stream()

        if init_fn is not None:
            self.resource = init_fn()

    def process(self, task):
        """
        execute stage func.
        """
        if self.use_multistream:
            with torch.cuda.stream(self.stream):
                return self.work_fn(self.resource, task)
        else:
            return self.work_fn(self.resource, task)


class WorkerThread:
    """
    Use new threads to execute pipeline stages.
    """

    def __init__(
        self,
        work_fn,
        init_fn,
        job_queue,
        result_queue,
        curr_worker_num,
        next_worker_num,
        use_multistream=False,
    ):
        self.worker = SimpleWorker(work_fn, init_fn, use_multistream=use_multistream)
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.curr_worker_num = curr_worker_num
        self.next_worker_num = next_worker_num
        self.worker_thread = None
        self.last_tid = None

    def run_worker(self, prof=None):
        """
        use worker to process stage func.
        """
        while True:
            task = self.get_from_job_queue()
            is_stop = isinstance(task, StopTask)

            if is_stop:
                result_task = task
            else:
                result = self.worker.process(task.value)
                result_task = Task(task.uid, result)

            self.put_to_result_queue(result_task)

            if prof is not None:
                prof.step()

            if is_stop:
                break

    def run(self):
        """
        start worker thread.
        """
        self.worker_thread = Thread(target=self.run_worker, args=())
        self.worker_thread.start()

    def wait_stop(self):
        """
        wait for worker thread to stop.
        """
        self.worker_thread.join()

    def put_stop_task(self):
        """
        put stop task to result queue.
        """
        self.result_queue.put(StopTask())

    def get_from_job_queue(self):
        """
        get task from job queue.
        """
        return self.job_queue.get()

    def put_to_result_queue(self, result):
        """
        put task to result queue.
        """
        self.result_queue.put(result)


class ProfileWorkerThread(WorkerThread):
    """
    To profile torch timeline in worker thread.
    """

    def __init__(
        self,
        work_fn,
        init_fn,
        job_queue,
        result_queue,
        curr_worker_num,
        next_worker_num,
        use_multistream=False,
    ):
        super().__init__(
            work_fn,
            init_fn,
            job_queue,
            result_queue,
            curr_worker_num,
            next_worker_num,
            use_multistream=use_multistream,
        )

    def run_worker(self):
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=120, warmup=120, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.expanduser("~/landmark/")),
            with_stack=True,
        ) as prof:
            super().run_worker(prof)

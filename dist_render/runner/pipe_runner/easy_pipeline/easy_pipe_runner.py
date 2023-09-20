from abc import abstractmethod
from queue import SimpleQueue
from uuid import uuid4

from .task import StopTask, Task


class EasyPipeRunner:
    """
    Pipeline runner implemented by easypipe.
    """

    def __init__(
        self,
    ) -> None:
        self.job_queue = SimpleQueue()
        self.pipe = self.create_pipe()
        self.pipe.start()
        self.result_queue = self.pipe.get_result_queue()

    @abstractmethod
    def create_pipe(self):
        pass

    def run_batch(self, single_pose):
        """
        put nerf pose to pipeline job queue.
        """
        uid = uuid4()
        self.job_queue.put(Task(uid, single_pose))
        return uid

    def stop(self):
        """
        stop pipeline
        """
        self.job_queue.put(StopTask())
        self.pipe.wait_stop()

    def get_from_result_queue(self):
        """
        get result from pipeline result queue.
        """
        return self.result_queue.get()

    def wait_stop(self):
        """
        wait stop task in result queue.
        """
        while True:
            result_task = self.get_from_result_queue()

            if isinstance(result_task, StopTask):
                break

    def get_result(self):
        """
        get result value from pipeline result queue.
        """
        while True:
            result_task = self.get_from_result_queue()

            if isinstance(result_task, StopTask):
                break

            return result_task.value

    def get_result_task(self, block=False):
        """
        get result task from pipeline result queue.
        """
        if block:
            return self.result_queue.get()
        return self.result_queue.get_nowait()

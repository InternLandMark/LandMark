import time
from enum import Enum

import torch


class ProfileStageType(Enum):
    """
    different stage type in pipeline to profile.
    """

    Preprocess = "Preprocess"
    Render = "Render"
    ModelInfer = "ModelInfer"
    Postprocess = "Postprocess"


class PipeStagesProfiler:
    """
    Record times of different pipeline stages.
    """

    start_record = {stage: None for stage in ProfileStageType}
    all_record = {stage: [] for stage in ProfileStageType}
    warm_up_len = 60
    flush_len = 400
    flushed = False
    module_profiler = None

    @classmethod
    def start(cls, stage: ProfileStageType):
        """
        record start time of the given stage.

        Args:
            stage(ProfileStageType): stage type.
        """
        torch.cuda.synchronize()
        cls.start_record[stage] = time.time()

    @classmethod
    def end(cls, stage: ProfileStageType):
        """
        record end time of the given stage.

        Args:
            stage(ProfileStageType): stage type.
        """
        assert cls.start_record[stage] is not None
        torch.cuda.synchronize()
        cls.all_record[stage].append(time.time() - cls.start_record[stage])
        cls.start_record[stage] = None
        if len(cls.all_record[stage]) > cls.warm_up_len:
            print(
                "Avg ",
                stage.value,
                " Time:",
                sum(cls.all_record[stage][cls.warm_up_len :]) / len(cls.all_record[stage][cls.warm_up_len :]),
                flush=True,
            )

        if (
            not cls.flushed
            and stage is ProfileStageType.Postprocess
            and cls.module_profiler is not None
            and len(cls.all_record[stage][cls.warm_up_len :]) > cls.flush_len
        ):
            cls.module_profiler.flush()
            cls.flushed = True


class TPCommunicationProfiler:
    """
    Record times of tensor parallel communication time and size
    """

    count = 0
    communication_time_one_render = 0
    communication_times = []
    communication_size_one_render = 0
    communication_size = []
    start_record = None
    warm_up_len = 60

    @classmethod
    def start(cls):
        """
        record start time of tp communication.
        """
        torch.cuda.synchronize()
        cls.start_record = time.time()

    @classmethod
    def convert_size(cls, size):
        """
        convert size to larger unit.
        """
        units = ["Bytes", "KB", "MB", "GB", "TB"]
        unit = units[0]
        for i in range(1, len(units)):
            if size >= 1024:
                size /= 1024
                unit = units[i]
            else:
                break
        return f"{size} {unit}"

    @classmethod
    def end(cls, tensor=None, com_times_per_infer=3):
        """
        record end time and size of tp communication.

        Args:
            tensor(Tensor): communication tensor.
            com_times_per_infer(int): communication times per iter.
        """
        torch.cuda.synchronize()
        assert cls.start_record is not None
        cls.communication_time_one_render += time.time() - cls.start_record
        if tensor is not None:
            cls.communication_size_one_render += tensor.element_size() * tensor.nelement()

        cls.start_record = None
        cls.count += 1
        if cls.count == com_times_per_infer:
            cls.communication_times.append(cls.communication_time_one_render)
            cls.communication_time_one_render = 0
            cls.communication_size.append(cls.communication_size_one_render)
            cls.communication_size_one_render = 0
            cls.count = 0
            if len(cls.communication_times) > cls.warm_up_len:
                print(
                    "Avg tp all gather time:",
                    sum(cls.communication_times[cls.warm_up_len :]) / len(cls.communication_times[cls.warm_up_len :]),
                    flush=True,
                )
            if len(cls.communication_size) > cls.warm_up_len:
                print(
                    "Avg tp all gather size:",
                    cls.convert_size(
                        sum(cls.communication_size[cls.warm_up_len :]) / len(cls.communication_size[cls.warm_up_len :])
                    ),
                    flush=True,
                )

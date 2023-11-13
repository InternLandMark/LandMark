from enum import Enum


class DatasetType(Enum):
    """
    Nerf dataset types.
    """

    City = "City"


class LoadStatus(Enum):
    """
    Used to control load status if use dynamic_fetching.
    """

    IDLE = 0  # doing nothing
    StartLoad = 1  # start to load, this status will send `LoadPlane` signal to all nodes' rank 0.
    Loading = 2  # during loading the plane from cpu to gpu
    LoadDone = 3  # load done
    Init = 5  # the first frame and no plane on gpu yet.


class ModelType(Enum):
    """
    Nerf mode types.
    """

    Torch = "Torch"
    MultiBlockTorch = "MultiBlockTorch"
    MultiBlockTensorParallelTorch = "MultiBlockTensorParallelTorch"
    TorchKernelFusion = "TorchKernelFusion"
    MultiBlockKernelFusion = "MultiBlockKernelFusion"
    MultiBlockTensorParallelKernelFusion = "MultiBlockTensorParallelKernelFusion"
    MovingAreaTorch = "MovingAreaTorch"


class RunnerType(Enum):
    """
    Runner type used in engine.
    """

    EasyPipeRunner = "EasyPipeRunner"
    DDPRunner = "DDPRunner"


class EngineType(Enum):
    """
    Different main rank engine type.
    """

    PipeEngine = "PipeEngine"
    DDPEngine = "DDPEngine"

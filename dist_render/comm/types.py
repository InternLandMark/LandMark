from enum import Enum

from dataset.city import CityAllBlockTestData, CityPartTestData
from ddp_infer.context import NerfContext
from ddp_infer.inferer_impl import (
    MultiBlockKernelFusionNerfDDPInfererImpl,
    MultiBlockTensorParallelKernelFusionNerfDDPInfererImpl,
    MultiBlockTensorParallelTorchNerfDDPInfererImpl,
    MultiBlockTorchNerfDDPInfererImpl,
    TorchNerfDDPInfererImpl,
)


class DatasetType(Enum):
    """
    Nerf dataset types.
    """

    CityPart = "CityPart"
    CityAllBlock = "CityAllBlock"


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


# TODO: build factory automatically.
def inferer_impl_factory(model_type: ModelType, profile_stages: bool = False):
    """
    Create inferer implementation instance according to the given model type.

    Args:
        model_type(ModelType): the model executor type/inferer impl type.
        profile_stages(bool): profile stages time cost.
    """
    context = NerfContext
    if model_type is ModelType.Torch:
        return TorchNerfDDPInfererImpl(context=context)
    elif model_type is ModelType.MultiBlockTorch:
        return MultiBlockTorchNerfDDPInfererImpl(context=context)
    elif model_type is ModelType.MultiBlockTensorParallelTorch:
        return MultiBlockTensorParallelTorchNerfDDPInfererImpl(context=context, profile_stages=profile_stages)
    elif model_type is ModelType.TorchKernelFusion:
        # return KernelFusionNerfDDPInfererImpl(context=KernelFusionNerfPipeContext())
        raise NotImplementedError()
    elif model_type is ModelType.MultiBlockKernelFusion:
        return MultiBlockKernelFusionNerfDDPInfererImpl(context=context)
    elif model_type is ModelType.MultiBlockTensorParallelKernelFusion:
        return MultiBlockTensorParallelKernelFusionNerfDDPInfererImpl(context=context)
    else:
        raise NotImplementedError()


def dataset_factory(dataset_type: DatasetType):
    """
    Create city dataset instance according to different type.

    Args:
        dataset_type(DatasetType): city dataset type.
    """
    if dataset_type is DatasetType.CityPart:
        return CityPartTestData(NerfContext.args_nerf)
    elif dataset_type is DatasetType.CityAllBlock:
        return CityAllBlockTestData(NerfContext.args_nerf)
    else:
        raise NotImplementedError()


def runner_factory(runner_type):
    """
    Create runner instance according to different type.

    Args:
        runner_type(RunnerType): main rank engine runner type.
    """
    if runner_type is RunnerType.EasyPipeRunner:
        from runner.pipe_runner.nerf_pipe_runner import NerfPipeRunner

        return NerfPipeRunner
    elif runner_type is RunnerType.DDPRunner:
        from runner.ddp_runner.nerf_ddp_runner import NerfDDPRunner

        return NerfDDPRunner
    else:
        raise NotImplementedError()

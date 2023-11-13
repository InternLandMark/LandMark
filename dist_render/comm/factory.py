from dist_render.comm.types import DatasetType, ModelType, RunnerType
from dist_render.dataset.city import CityTestData
from dist_render.ddp_infer.context import NerfContext
from dist_render.ddp_infer.inferer_impl import (
    MovingAreaTorchNerfDDPInfererImpl,
    MultiBlockKernelFusionNerfDDPInfererImpl,
    MultiBlockTensorParallelKernelFusionNerfDDPInfererImpl,
    MultiBlockTensorParallelTorchNerfDDPInfererImpl,
    MultiBlockTorchNerfDDPInfererImpl,
    TorchNerfDDPInfererImpl,
)


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
    elif model_type is ModelType.MovingAreaTorch:
        return MovingAreaTorchNerfDDPInfererImpl(context=context)
    else:
        raise NotImplementedError()


def dataset_factory(dataset_type: DatasetType):
    """
    Create city dataset instance according to different type.

    Args:
        dataset_type(DatasetType): city dataset type.
    """
    if dataset_type is DatasetType.City:
        return CityTestData(NerfContext.args_nerf)
    else:
        raise NotImplementedError()


def runner_factory(runner_type):
    """
    Create runner instance according to different type.

    Args:
        runner_type(RunnerType): main rank engine runner type.
    """
    if runner_type is RunnerType.EasyPipeRunner:
        from dist_render.runner.pipe_runner.nerf_pipe_runner import NerfPipeRunner

        return NerfPipeRunner
    elif runner_type is RunnerType.DDPRunner:
        from dist_render.runner.ddp_runner.nerf_ddp_runner import NerfDDPRunner

        return NerfDDPRunner
    else:
        raise NotImplementedError()

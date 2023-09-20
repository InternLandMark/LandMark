import os

import configargparse
import numpy as np
import torch
from comm.env import EnvSetting
from comm.types import DatasetType, EngineType, ModelType
from ddp_infer.config_parser import parse_nerf_config_args
from ddp_infer.context import NerfContext
from ddp_infer.nerf_inferer import GlobalArgsManager
from engine.ddp_engine import MainRankDDPRenderEngine
from engine.pipe_engine import MainRankPipeRenderEngine
from engine.render_engine import OtherRankRenderEngine


def launch(
    engine_type=EngineType.PipeEngine,
    model_type=ModelType.Torch,
    dataset=DatasetType.CityPart,
    use_multistream=False,
    test=False,
    save_png=False,
    pose_num=-1,
):
    """
    launch master and worker on distributed cluster.

    Args:
        engine_type(EngineType): main rank engine type.
        model_type(ModelType): the executor type you want to run your model on.
        dataset(DatasetType): part dataset or all dataset.
        use_multistream(bool): whether to use multistream optimization.
        test(bool): whether to run test dataset after launching the engine.
        save_png(bool): whether to save png picture when engine run test dataset.
        pose_num(int): the pose number you want to run in test dataset.
    """
    engine = None
    if EnvSetting.RANK == 0:
        if engine_type is EngineType.PipeEngine:
            engine = MainRankPipeRenderEngine(
                dataset=dataset,
                model_type=model_type,
                use_multistream=use_multistream,
                save_png=save_png,
                pose_num=pose_num,
            )
        elif engine_type is EngineType.DDPEngine:
            engine = MainRankDDPRenderEngine(dataset=dataset, model_type=model_type, pose_num=pose_num)
    else:
        engine = OtherRankRenderEngine(dataset=dataset, model_type=model_type, use_multistream=use_multistream)

    engine.start(test=test)


def change_mode(mode: str):
    """
    change nerf encode_app code or city editing mode.

    Args:
        mode(str): mode str.
    """
    print(f"render mode: {mode}")

    if mode == "sunset":
        GlobalArgsManager().set_arg("app_code", 2180)
    elif mode == "sunshine":
        GlobalArgsManager().set_arg("app_code", 1550)
    elif mode == "cloudy":
        GlobalArgsManager().set_arg("app_code", 2940)
    elif mode == "fog":
        GlobalArgsManager().set_arg("app_code", 1820)
    elif mode == "reset":
        GlobalArgsManager().set_arg("app_code", 2930)
        GlobalArgsManager().set_arg("edit_mode", 0)
    elif mode == "resetBuild":
        GlobalArgsManager().set_arg("edit_mode", 0)
    elif mode == "newBuild":
        GlobalArgsManager().set_arg("edit_mode", 1)
    elif mode == "removeBuild":
        GlobalArgsManager().set_arg("edit_mode", 2)
    else:
        assert False, f"No such mode :{mode}"


def adjust_nerf_context(config_path, model_path, aabb, render_batch_size, alpha_mask_filter_thre):
    """
    adjust nerf context attrs according to configs parse from user.

    Args:
        config_path(str): model config path.
        model_path(str): ckpt model path.
        aabb(str): the corresponding aabb value when you use part model.
        render_batch_size(int): nerf chunk size.
        alpha_mask_filter_thre(float): nerf alpha mask filter threshold.
    """
    assert config_path is not None
    assert os.path.exists(config_path)
    NerfContext.args_nerf = parse_nerf_config_args(cmd=f"--config {config_path}")

    if model_path is not None:
        assert os.path.exists(model_path)
        NerfContext.model_path = model_path
    elif model_path is None and NerfContext.model_path is None:
        assert os.path.exists(NerfContext.args_nerf.ckpt)
        NerfContext.model_path = NerfContext.args_nerf.ckpt

    if aabb is not None:
        aabb = np.array(eval(aabb))  # pylint: disable=W0123
        NerfContext.aabb = torch.from_numpy(aabb).float()

    if render_batch_size is not None:
        NerfContext.render_batch_size = render_batch_size

    if alpha_mask_filter_thre is not None:
        NerfContext.alpha_mask_filter_thre = alpha_mask_filter_thre

    assert NerfContext.args_nerf
    assert NerfContext.model_path

    NerfContext.args_nerf.ckpt = NerfContext.model_path
    to_cover = ["half_precision_param", "sampling_opt", "render_batch_size", "alpha_mask_filter_thre"]
    for name in to_cover:
        if hasattr(NerfContext, name) and getattr(NerfContext, name) is not None:
            setattr(NerfContext.args_nerf, name, getattr(NerfContext, name))


def engine_config_parser():
    """
    parse engine configs from user.
    """
    parser = configargparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="CityPart")
    parser.add_argument("--model_type", type=str, default="MultiBlock4KTensorParallelEncAppTorch")
    parser.add_argument("--use_multistream", default=False, action="store_true")
    parser.add_argument("--test", default=True, action="store_true")
    parser.add_argument("--save_png", default=False, action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--pose_num", type=int, default=-1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--aabb", type=str, default=None)
    parser.add_argument("--render_batch_size", type=int, default=None)
    parser.add_argument("--alpha_mask_filter_thre", type=float, default=None)

    return parser.parse_args()


def check_model_type(m_type):
    """
    check whether the executor type is right.

    Args:
        m_type(ModelType): the executor type.
    """
    if m_type in [
        ModelType.MultiBlockTorch,
        ModelType.MultiBlockKernelFusion,
        ModelType.MultiBlockTensorParallelTorch,
        ModelType.MultiBlockTensorParallelKernelFusion,
    ]:
        assert NerfContext.args_nerf.branch_parallel, "The model should be trained by branch parallel"
    elif m_type in [ModelType.Torch, ModelType.TorchKernelFusion]:
        assert not NerfContext.args_nerf.branch_parallel, "The model should not be trained by branch parallel"


if __name__ == "__main__":
    engine_args = engine_config_parser()
    adjust_nerf_context(
        engine_args.config,
        engine_args.ckpt,
        engine_args.aabb,
        engine_args.render_batch_size,
        engine_args.alpha_mask_filter_thre,
    )
    check_model_type(ModelType(engine_args.model_type))
    launch(
        dataset=DatasetType(engine_args.dataset),
        model_type=ModelType(engine_args.model_type),
        use_multistream=engine_args.use_multistream,
        test=engine_args.test,
        save_png=engine_args.save_png,
        pose_num=engine_args.pose_num,
    )

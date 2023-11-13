from functools import partial

import numpy as np

from dist_render.comm.parallel_context import (
    ParallelContext,
    ParallelGroup,
    init_parallel_context,
)
from dist_render.comm.types import ModelType
from dist_render.ddp_infer.nerf_inferer import (
    generate_hw,
    infer,
    model_infer_postprocess,
    model_infer_preprocess,
)
from dist_render.runner.pipe_runner.easy_pipeline.easy_pipe_runner import EasyPipeRunner
from dist_render.runner.pipe_runner.easy_pipeline.pipeline import (
    PipelineItem,
    SimplePipeline,
)


def transform_pose_to_rays_item(
    init_res,
    pose: np.ndarray,
    model_type: ModelType,
    data_parallel_local_rank: int,
    rank: int,
    data_parallel_group_world_size: int,
):
    """
    model preprocess stage item func

    Args:
        init_res(tuple): image height and width from init func.
        pose(ndarray): nerf single pose.
        model_type(ModelType): the model executor type.
        data_parallel_local_rank(int): local rank of data parallel.
        rank(int): global rank.
        data_parallel_group_world_size(int): world size of data parallel group.

    Returns:
        tuple: preprocess results.
    """
    return model_infer_preprocess(
        pose,
        H=init_res[0],
        W=init_res[1],
        model_type=model_type,
        data_parallel_local_rank=data_parallel_local_rank,
        rank=rank,
        data_parallel_group_world_size=data_parallel_group_world_size,
    )


def render_item(init_res, batch_rays: np.ndarray, model_type: ModelType):
    """
    model render stage item func

    Args:
        init_res(tuple): image height and width from init func.
        batch_rays(ndarray): nerf rays generated from pose.
        model_type(ModelType): the model executor type.

    Returns:
        tuple: render results.
    """
    # Any operation of rays before inference should be done in previous stage.
    return infer(batch_rays, H=init_res[0], W=init_res[1], model_type=model_type)


def transform_output_to_image(init_res, batch_results: np.ndarray, model_type: ModelType):
    """
    model postprocess stage item func

    Args:
        init_res(tuple): image height and width from init func.
        batch_results(ndarray): nerf rendering result.
        model_type(ModelType): the model executor type.

    Returns:
        ndarray: numpy image result.
    """
    return model_infer_postprocess(batch_results, H=init_res[0], W=init_res[1], model_type=model_type)


class NerfPipeRunner(EasyPipeRunner):
    """
    Pipeline runner of nerf model.
    """

    def __init__(self, dataset_type, model_type, use_multistream=False, profile_timeline=False) -> None:
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.use_multistream = use_multistream
        self.profile_timeline = profile_timeline
        init_parallel_context()
        # Super init will call create_pipe
        super().__init__()

    def create_pipe(
        self,
    ):
        """
        use different model stage items to create pipeline
        """
        generate_hw_with_dataset = partial(generate_hw, self.dataset_type)
        pipeline_items = [
            PipelineItem(
                partial(
                    transform_pose_to_rays_item,
                    model_type=self.model_type,
                    data_parallel_local_rank=ParallelContext().get_local_rank(ParallelGroup.ProcessesInDataParalGroup),
                    rank=ParallelContext().get_rank(),
                    data_parallel_group_world_size=ParallelContext().get_group_world_size(
                        ParallelGroup.ProcessesInDataParalGroup
                    ),
                ),
                generate_hw_with_dataset,
                1,
                -1,
                use_multistream=self.use_multistream,
            ),
            PipelineItem(
                partial(render_item, model_type=self.model_type),
                generate_hw_with_dataset,
                1,
                -1,
                profile_timeline=self.profile_timeline,
            ),
            PipelineItem(
                partial(transform_output_to_image, model_type=self.model_type),
                generate_hw_with_dataset,
                1,
                -1,
                use_multistream=self.use_multistream,
            ),
        ]
        return SimplePipeline(pipeline_items, self.job_queue)

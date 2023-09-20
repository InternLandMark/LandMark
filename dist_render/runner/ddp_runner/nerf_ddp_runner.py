from comm.parallel_context import ParallelContext, ParallelGroup, init_parallel_context
from ddp_infer.nerf_inferer import (
    generate_hw,
    infer,
    model_infer_postprocess,
    model_infer_preprocess,
)


class NerfDDPRunner:
    """
    DDP runner.
    """

    def __init__(self, dataset_type, model_type) -> None:
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.H, self.W = generate_hw(self.dataset_type)
        init_parallel_context()

    def run_batch(self, single_pose):
        """
        run pose without pipeline.
        """
        preprocess_res = model_infer_preprocess(
            single_pose,
            H=self.H,
            W=self.W,
            model_type=self.model_type,
            data_parallel_local_rank=ParallelContext().get_local_rank(ParallelGroup.ProcessesInDataParalGroup),
            rank=ParallelContext().get_rank(),
            data_parallel_group_world_size=ParallelContext().get_group_world_size(
                ParallelGroup.ProcessesInDataParalGroup
            ),
            buffer_len=1,
        )
        infer_res = infer(preprocess_res, H=self.H, W=self.W, model_type=self.model_type)
        return model_infer_postprocess(infer_res, H=self.H, W=self.W, model_type=self.model_type)

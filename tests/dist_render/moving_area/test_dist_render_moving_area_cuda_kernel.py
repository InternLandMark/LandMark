import torch.distributed as dist

from dist_render.comm.types import DatasetType, ModelType
from dist_render.renderer import (
    adjust_nerf_context,
    check_model_type,
    engine_config_parser,
    launch,
)
from tests import P_CLUSTER_CKPT_DIR, P_CLUSTER_SHCITY_DATAROOT
from tests.utils import dist_render_setting, get_pictures_psnr, render_check


class TestDistRenderCityMovingAreaCudaKernel:
    """
    Integrite test for dist_render_city_moving_area_encodeapp_cuda_kernel rendering
    """

    def setup_class(self):

        engine_cmd = "--config "
        self.conf_dir = "confs/dist_render_conf/dist_render_city_moving_area_encodeapp.txt "
        engine_cmd += self.conf_dir
        engine_cmd += "--ckpt "
        self.ckpt_dir = (
            P_CLUSTER_CKPT_DIR
            + "large_ckpts/2_almostall_1k_bp_div8x6_encodeapp_run2_hull_train2k_hull256_nostratefied/"
            + "2_almostall_1k_bp_div8x6_encodeapp_run2_hull_train2k_hull256_nostratefied-merged-stack.th "
        )
        engine_cmd += self.ckpt_dir
        engine_cmd += "--model_type=MovingAreaCudaKernel "
        engine_cmd += dist_render_setting()
        self.engine_args = engine_config_parser(engine_cmd)
        adjust_nerf_context(
            self.engine_args.config,
            self.engine_args.ckpt,
            self.engine_args.aabb,
            self.engine_args.render_batch_size,
            self.engine_args.alpha_mask_filter_thre,
            P_CLUSTER_SHCITY_DATAROOT,
        )
        check_model_type(ModelType(self.engine_args.model_type))

    def teardown_class(self):
        dist.destroy_process_group()

    def test_render(self):
        launch(
            dataset=DatasetType(self.engine_args.dataset),
            model_type=ModelType(self.engine_args.model_type),
            use_multistream=self.engine_args.use_multistream,
            test=self.engine_args.test,
            save_png=self.engine_args.save_png,
            pose_num=self.engine_args.pose_num,
        )
        psnr = get_pictures_psnr()
        render_check(new_psnr=psnr, base_psnr=16.703250, class_name=self.__class__.__name__)

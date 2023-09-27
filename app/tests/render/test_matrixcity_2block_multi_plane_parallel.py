from app.renderer import init_render_env, render
from app.tests import P_CLUSTER_CKPT_DIR
from app.tests.utils import render_check, render_setting


class TestMatrixcity2BlockMultiPlaneParallel:
    """
    Integrite test for Matrixcity2BlockMultiPlaneParallel rendering
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_multi_plane_parallel.txt "
        cmd += self.conf_dir
        cmd += "--ckpt "
        self.ckpt_dir = (
            P_CLUSTER_CKPT_DIR + "matrix_city_block_1+2_plane_parallel/matrix_city_block_1+2_plane_parallel-merged.th "
        )
        cmd += self.ckpt_dir
        cmd += render_setting()
        self.args = init_render_env(cmd)

    def teardown_class(self):
        pass

    def test_render(self):
        psnr = render(self.args)
        render_check(new_psnr=psnr, base_psnr=25.978542, config=self.conf_dir, ckpt=self.ckpt_dir)

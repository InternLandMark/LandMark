from app.renderer import render
from app.tests import P_CLUSTER_CKPT_DIR
from app.tests.render.utils import check, ci_setting, init_render_env


class TestMatrixcity2BlockMulti:
    """
    Integrite test for Matrixcity2BlockMulti rendering
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_multi.txt "
        cmd += self.conf_dir
        cmd += "--ckpt "
        self.ckpt_dir = P_CLUSTER_CKPT_DIR + "matrix_city_block_1+2_multi/matrix_city_block_1+2_multi.th "
        cmd += self.ckpt_dir
        cmd += ci_setting()
        self.args = init_render_env(cmd)

    def teardown_class(self):
        pass

    def test_render(self):
        psnr = render(self.args)
        check(new_psnr=psnr, base_psnr=25.401024, config=self.conf_dir, ckpt=self.ckpt_dir)

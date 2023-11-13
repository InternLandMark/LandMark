from app.renderer import init_render_env, render
from tests import P_CLUSTER_CKPT_DIR
from tests.utils import render_check, render_setting


class TestMatrixcity2BlockLowquality:
    """
    Integrite test for Matrixcity2BlockLowquality rendering
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_lowquality.txt "
        cmd += self.conf_dir
        cmd += "--ckpt "
        self.ckpt_dir = (
            P_CLUSTER_CKPT_DIR + "matrix_city/matrix_city_block_1+2_lowquality/matrix_city_block_1+2_lowquality.th "
        )
        cmd += self.ckpt_dir
        cmd += render_setting()
        self.args = init_render_env(cmd)

    def teardown_class(self):
        pass

    def test_render(self):
        psnr = render(self.args)
        render_check(new_psnr=psnr, base_psnr=22.908665, class_name=self.__class__.__name__)

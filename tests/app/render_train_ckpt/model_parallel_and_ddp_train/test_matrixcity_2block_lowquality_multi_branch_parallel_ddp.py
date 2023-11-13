from app.renderer import init_render_env, render
from tests.utils import render_check, render_setting


class TestMatrixcity2BlockMultiBranchParallelDDPLowquality:
    """
    Integrite test for Matrixcity2BlockMultiLowquality rendering with DDP and branch parallel
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_lowquality_multi_branch_parallel_ddp.txt "
        cmd += self.conf_dir
        cmd += "--ckpt "
        self.ckpt_dir = "auto "
        cmd += self.ckpt_dir
        cmd += render_setting()
        self.args = init_render_env(cmd)

    def teardown_class(self):
        pass

    def test_render(self):
        psnr = render(self.args)
        render_check(new_psnr=psnr, base_psnr=22.908665, class_name=self.__class__.__name__)

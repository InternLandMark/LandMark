from app.renderer import init_render_env, render
from tests import P_CLUSTER_CKPT_DIR
from tests.utils import render_check, render_setting


class TestMatrixcity2BlockMultiBranchParallel:
    """
    Integrite test for Matrixcity2BlockMultiBranchParallel rendering
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_multi_branch_parallel.txt "
        cmd += self.conf_dir
        cmd += "--ckpt "
        self.ckpt_dir = (
            P_CLUSTER_CKPT_DIR
            + "matrix_city/matrix_city_block_1+2_branch_parallel/matrix_city_block_1+2_branch_parallel-merged-stack.th "
        )
        cmd += self.ckpt_dir
        cmd += "--ckpt_type full "
        cmd += render_setting()
        self.args = init_render_env(cmd)

    def teardown_class(self):
        pass

    def test_render(self):
        psnr = render(self.args)
        render_check(new_psnr=psnr, base_psnr=26.355559, class_name=self.__class__.__name__)

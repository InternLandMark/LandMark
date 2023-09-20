from app.tests.train.utils import check, ci_setting, init_train_env
from app.trainer import train


class TestMatrixcity2BlockLowquality:
    """
    Integrite test for Matrixcity2BlockLowquality training
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_lowquality.txt "
        cmd += self.conf_dir
        cmd += ci_setting()
        self.args = init_train_env(cmd)

    def teardown_class(self):
        pass

    def test_render(self):
        psnr = train(self.args)
        check(new_psnr=psnr, base_psnr=22.908665, config=self.conf_dir)

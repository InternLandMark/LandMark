import torch.distributed as dist

from app.tests.utils import train_check, train_setting
from app.trainer import init_train_env, train


class TestMatrixcity2BlockMultiDDPLowquality:
    """
    Integrite test for Matrixcity2BlockLowquality training with DDP
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_lowquality.txt "
        cmd += self.conf_dir
        cmd += train_setting()
        self.args = init_train_env(cmd)

    def teardown_class(self):
        dist.destroy_process_group()

    def test_train(self):
        psnr = train(self.args)
        train_check(new_psnr=psnr, base_psnr=22.908665, config=self.conf_dir)

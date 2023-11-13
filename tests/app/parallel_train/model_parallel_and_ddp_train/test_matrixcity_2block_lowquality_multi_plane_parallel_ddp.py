import torch.distributed as dist

from app.trainer import init_train_env, train
from tests.utils import train_check, train_setting


class TestMatrixcity2BlockMultiPlaneParallelDDPLowquality:
    """
    Integrite test for Matrixcity2BlockMultiLowquality training with DDP and plane parallel
    """

    def setup_class(self):
        cmd = "--config "
        self.conf_dir = "confs/matrixcity/matrixcity_2block_lowquality_multi_plane_parallel_ddp.txt "
        cmd += self.conf_dir
        cmd += train_setting()
        self.args = init_train_env(cmd)

    def teardown_class(self):
        dist.destroy_process_group()

    def test_train(self):
        psnr = train(self.args)
        train_check(new_psnr=psnr, base_psnr=22.908665, class_name=self.__class__.__name__)

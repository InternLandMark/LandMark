import os

from app.tests import P_CLUSTER_DATAROOT

MASTER_PORT = "29510"


def render_check(new_psnr, base_psnr, value=0.05, config="", ckpt=""):
    if new_psnr > base_psnr:
        pass
    else:
        assert base_psnr / new_psnr - 1 <= value, (
            f"Config [{config}] and ckpt [{ckpt}] rendering verification failed. "
            f"Running result: [{new_psnr}] // Baseline: [{base_psnr}]"
        )


def render_setting():
    global MASTER_PORT
    if int(MASTER_PORT) > 29700:
        MASTER_PORT = "29510"
    else:
        MASTER_PORT = str(int(MASTER_PORT) + 10)
    os.environ["MASTER_PORT"] = MASTER_PORT

    cmd = ""
    cmd += "--dataroot " + P_CLUSTER_DATAROOT
    cmd += "--compute_extra_metrics 0 "
    cmd += "--skip_save_imgs 1 "
    cmd += "--env slurm "
    return cmd


def train_check(new_psnr, base_psnr, value=0.05, config=""):
    if new_psnr > base_psnr:
        pass
    else:
        assert (
            base_psnr / new_psnr - 1 <= value
        ), f"Config [{config}] training verification failed. Running result: [{new_psnr}] // Baseline: [{base_psnr}]"


def train_setting():
    global MASTER_PORT
    cmd = ""
    cmd += "--dataroot " + P_CLUSTER_DATAROOT
    cmd += "--env slurm "
    if int(MASTER_PORT) > 29700:
        MASTER_PORT = "29510"
    else:
        MASTER_PORT = str(int(MASTER_PORT) + 10)
    os.environ["MASTER_PORT"] = MASTER_PORT
    return cmd

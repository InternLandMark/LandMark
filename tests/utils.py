import json
import os

import cv2
import numpy as np
import torch

from dist_render.comm.env import EnvSetting
from tests import P_CLUSTER_MATRIXCITY_DATAROOT, P_CLUSTER_SHCITY_DATAROOT

MASTER_PORT = "29400"


def render_check(new_psnr, base_psnr, value=0.05, class_name=""):
    if new_psnr > base_psnr:
        pass
    else:
        assert (
            base_psnr / new_psnr - 1 <= value
        ), f"Test [{class_name}] rendering verification failed. Running result: [{new_psnr}] // Baseline: [{base_psnr}]"


def render_setting(dataset="matrixcity"):
    global MASTER_PORT
    if int(MASTER_PORT) > 29600:
        MASTER_PORT = "29400"
    else:
        MASTER_PORT = str(int(MASTER_PORT) + 10)
    os.environ["MASTER_PORT"] = MASTER_PORT

    if dataset == "shcity":
        dataroot = P_CLUSTER_SHCITY_DATAROOT
    else:
        dataroot = P_CLUSTER_MATRIXCITY_DATAROOT

    cmd = ""
    cmd += "--dataroot " + dataroot
    cmd += "--compute_extra_metrics 0 "
    cmd += "--skip_save_imgs 1 "
    cmd += "--env slurm "
    return cmd


def dist_render_setting():
    global MASTER_PORT
    if int(MASTER_PORT) > 29700:
        MASTER_PORT = "29510"
    else:
        MASTER_PORT = str(int(MASTER_PORT) + 10)
    os.environ["MASTER_PORT"] = MASTER_PORT
    EnvSetting.CI_TEST_PICTURES = 160

    cmd = "--save_png "
    return cmd


def train_check(new_psnr, base_psnr, value=0.05, class_name=""):
    if new_psnr > base_psnr:
        pass
    else:
        assert (
            base_psnr / new_psnr - 1 <= value
        ), f"Test [{class_name}] training verification failed. Running result: [{new_psnr}] // Baseline: [{base_psnr}]"


def train_setting():
    global MASTER_PORT
    cmd = ""
    cmd += "--dataroot " + P_CLUSTER_MATRIXCITY_DATAROOT
    cmd += "--env slurm "
    if int(MASTER_PORT) > 29700:
        MASTER_PORT = "29510"
    else:
        MASTER_PORT = str(int(MASTER_PORT) + 10)
    os.environ["MASTER_PORT"] = MASTER_PORT
    return cmd


def get_pictures_psnr():
    torch.distributed.barrier()
    PSNRs = []
    with open(
        "/mnt/petrelfs/share_data/landmarks/train_data/base/sh_city/2/almost_all/transforms_test.json",
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)
    fnames = [frame["file_path"] for frame in meta["frames"]]
    assert len(fnames) == 100
    for i in range(len(fnames)):
        dir_path = "./dist_render/picture"
        pic = cv2.cvtColor(cv2.imread(os.path.join(dir_path, f"{i}.png")), cv2.COLOR_BGR2RGB)
        rgb_map = torch.from_numpy(pic).to(torch.float32) / 255.0  # .view(H, W, 3)
        gt_pic = cv2.cvtColor(cv2.imread(fnames[i]), cv2.COLOR_BGR2RGB)
        gt_rgb = torch.from_numpy(gt_pic).to(torch.float32) / 255.0  # .view(H, W, 3)
        loss = torch.mean((rgb_map - gt_rgb) ** 2)
        PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
    psnr = np.mean(np.asarray(PSNRs))
    return psnr

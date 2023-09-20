import numpy as np
import torch
from tools.config_parser import ArgsParser
from tools.configs import ArgsConfig

from app.tests import P_CLUSTER_DATAROOT


def init_render_env(cmd):
    args_parser = ArgsParser(cmd)
    exp_args = args_parser.get_exp_args()
    model_args = args_parser.get_model_args()
    render_args = args_parser.get_render_args()

    args = ArgsConfig([exp_args, model_args, render_args])

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    args.device = device

    args.rank = 0
    args.world_size = 1

    return args


def check(new_psnr, base_psnr, value=0.05, config="", ckpt=""):
    if new_psnr > base_psnr:
        pass
    else:
        assert base_psnr / new_psnr - 1 <= value, (
            f"Config [{config}] and ckpt [{ckpt}] rendering verification failed. "
            f"Running result: [{new_psnr}] // Baseline: [{base_psnr}]"
        )


def ci_setting():
    cmd = ""
    cmd += "--dataroot " + P_CLUSTER_DATAROOT
    cmd += "--compute_extra_metrics 0 "
    cmd += "--skip_save_imgs 1 "

    return cmd

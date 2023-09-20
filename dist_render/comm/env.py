import os
import subprocess
from dataclasses import dataclass


def parse_env(env_name):
    """
    parse environment variable.

    Args:
        env_name(str): environment variable name.
    """
    value = os.getenv(env_name)
    if value in ["True", "ON", "1"]:
        return True
    elif value in [None, "False", "OFF", "0"]:
        return False
    else:
        raise NotImplementedError()


@dataclass
class EnvSetting:
    """Environment variable"""

    # Profile pipeline stages time cost.
    PROFILE_STAGES = parse_env("PROFILE_STAGES")
    # Profile main rank torch timeline, it will save timeline file in `~/landmark/` directory
    PROFILE_TIMELINE = parse_env("PROFILE_TIMELINE")
    # Render 1080p image
    RENDER_1080P = parse_env("RENDER_1080P")
    # Tensor parallel group world size
    TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", 1))
    HALF_PRECISION_PARAM = parse_env("HALF_PRECISION_PARAM")
    # Images saving path
    PNG_SAVING_PATH = os.environ.get("PNG_SAVING_PATH", None)
    # For per process in one node, load large scale ckpt to cpu memory orldly in case of OOM
    LOAD_ORDERLY = parse_env("LOAD_ORDERLY")
    # Open city block edition
    ENABLE_EDIT_MODE = parse_env("ENABLE_EDIT_MODE")
    SLURM = "SLURM_PROCID" in os.environ
    NCCL_TIMEOUT = int(os.environ.get("NCCL_TIMEOUT", 1800))

    if SLURM:
        # slurm cluster
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")  # 29500
        os.environ["MASTER_ADDR"] = subprocess.getoutput(
            f"scontrol show hostname {os.environ['SLURM_NODELIST']} | head -n1"
        )
        WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
        RANK = int(os.environ["SLURM_PROCID"])
    else:
        # aliyun cluster
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE"))
        RANK = int(os.environ.get("RANK"))

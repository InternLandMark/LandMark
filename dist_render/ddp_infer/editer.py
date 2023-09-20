import os
from enum import Enum

import numpy as np
import torch


class EditType(Enum):
    """
    block edition model type.
    """

    ResetBuild = 0
    NewBuild = 1
    RemoveBuild = 2


class Editer:
    """
    Edit single block of model
    """

    def __init__(self) -> None:
        self._edit_params = {
            EditType.ResetBuild: {"alpha_volume": None, "state_dict": None},
            EditType.NewBuild: {"alpha_volume": None, "state_dict": None},
            EditType.RemoveBuild: {"alpha_volume": None, "state_dict": None},
        }
        self._cur_edit_mode = EditType.ResetBuild

    def load_edit_part(self, model_path):
        """
        load block model of edition part

        Args:
            model_path(str): the whole model path.
        """
        # TODO remove hard code
        block_fp = model_path.replace("merged-stack", "sub15")
        block_ckpt = torch.load(block_fp, map_location="cpu")
        self._edit_params[EditType.ResetBuild]["state_dict"] = block_ckpt["state_dict"]
        alphaMask_length = np.prod(block_ckpt["alphaMask.shape"])
        alphaMask_shape = block_ckpt["alphaMask.shape"]
        alpha_volume_fp = os.path.join(os.path.dirname(model_path), "alpha_volume_clean.npy")
        self._edit_params[EditType.ResetBuild]["alpha_volume"] = torch.from_numpy(
            np.load(alpha_volume_fp)[:alphaMask_length].reshape(alphaMask_shape)
        )

        # load newBuild
        block_fp = model_path.replace("merged-stack", "sub15_add")
        self._edit_params[EditType.NewBuild]["state_dict"] = torch.load(block_fp, map_location="cpu")["state_dict"]
        alpha_volume_fp = os.path.join(os.path.dirname(model_path), "alpha_volume_add.npy")
        self._edit_params[EditType.NewBuild]["alpha_volume"] = torch.from_numpy(
            np.load(alpha_volume_fp)[:alphaMask_length].reshape(alphaMask_shape)
        )

        # load removeBuild
        block_fp = model_path.replace("merged-stack", "sub15_remove")
        self._edit_params[EditType.RemoveBuild]["state_dict"] = torch.load(block_fp, map_location="cpu")["state_dict"]
        alpha_volume_fp = os.path.join(os.path.dirname(model_path), "alpha_volume_remove.npy")
        self._edit_params[EditType.RemoveBuild]["alpha_volume"] = torch.from_numpy(
            np.load(alpha_volume_fp)[:alphaMask_length].reshape(alphaMask_shape)
        )

        print("Load edited block successfully.")

    def edit_model(self, edit_mode=0, model=None, device=None):
        """
        replace some whole model parameter with part model.

        Args:
            edit_mode(int): edition type.
            model(Module): Whole model.
            device(str): cuda device.
        """
        edit_mode = EditType(edit_mode)

        if self._cur_edit_mode is edit_mode:
            return

        # change block ckpt
        plane_names = ["density_plane", "app_plane"]
        line_names = ["density_line", "app_line"]
        for name, param in model.named_parameters():
            if any(plane_name in name for plane_name in plane_names):
                param.data[:, :, 15] = self._edit_params[edit_mode]["state_dict"][name].to(device)
            if any(line_name in name for line_name in line_names):
                param.data[..., [15]] = self._edit_params[edit_mode]["state_dict"][name].to(device)

        model.alphaMask.alpha_volume = self._edit_params[edit_mode]["alpha_volume"].float().to(device)

        self._cur_edit_mode = edit_mode

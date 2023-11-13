from dataclasses import dataclass

from dist_render.comm.env import EnvSetting


@dataclass
class NerfContext:
    """
    Nerf ddp render info
    """

    args_nerf = None
    png_saving_path = "./dist_render/picture/" if EnvSetting.PNG_SAVING_PATH is None else EnvSetting.PNG_SAVING_PATH
    half_precision_param = EnvSetting.HALF_PRECISION_PARAM
    load_orderly = EnvSetting.LOAD_ORDERLY
    enable_edit_mode = EnvSetting.ENABLE_EDIT_MODE
    model_path = None
    aabb = None  # torch.from_numpy(np.array([[-19.2500, -8.5000, -1.0000], [-2.7500, 8.5000, 2.0000]])).float()
    sampling_opt = 1

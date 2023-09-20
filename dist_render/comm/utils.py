import torch


def kwargs_tensors_to_device(kwargs, device):
    """
    move the tensors in kwargs to target device

    Args:
        kwargs(dict): nerf module kwargs.
        device(str): cuda device.
    """
    for key, value in kwargs.items():
        if isinstance(value, dict):
            kwargs_tensors_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            kwargs[key] = value.to(device)


def rm_ddp_prefix_in_state_dict_if_present(state_dict, prefix=".module"):
    """
    rm ".module" if exists, since that rendering will never use ddp to wrap modules
    warning: may cause bug if some of the modules exactly have the name that is composed of 'module'

    Args:
        state_dict(dict): model ckpt state dict.
    """
    keys = sorted(state_dict.keys())
    for k in keys:
        name = k.replace(prefix, "")
        state_dict[name] = state_dict.pop(k)

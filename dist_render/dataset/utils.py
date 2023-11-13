import numpy as np
import torch
from kornia import create_meshgrid

from app.tools.dataloader.ray_utils import load_json_drone_data


def read_poses_city(root_dir, image_scale, subfolder):
    """
    read poses from city datasets.

    Args:
        root_dir(str): dataset root directory path.
        image_scale(float): image scale ratio.
        subfolder(str): dataset sub directory path.
    Returns:
        list: poses.
    """
    meta = load_json_drone_data(root_dir, "test", image_scale, subfolder)
    poses, focals, _, hw, _ = meta.values()
    new_poses = []
    for idx, pose in enumerate(poses):
        new_pose = np.zeros((3, 5))
        new_pose[:3, :4] = pose[:3, :4]
        if len(focals) == 1:
            new_pose[:, -1] = focals[0]
        else:
            new_pose[:, -1] = focals[idx]
        new_poses.append(new_pose)
    return new_poses, hw


def get_ray_directions_blender(H, W, focal, center=None, device=torch.device("cuda")):
    """
    Generate blender ray directions.

    Args:
        H(int): image height.
        W(int): image width.
        focal(tuple): focal value.
        center(Union[List, None]): center point.

    Returns:
        Tensor: ray directions.
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)[0]  # +0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [((i - cent[0]) / focal[0]).cuda(), (-(j - cent[1]) / focal[1]).cuda(), -torch.ones_like(i, device="cuda:0")],
        -1,
    )  # (H, W, 3)
    # directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions


def get_ray_directions_blender_faster(H, W, focal, device=torch.device("cuda")):
    constcent = [W / 2, H / 2]

    directions = torch.empty((3, H, W), dtype=torch.float32, device=device)
    directions[0, :, :] = torch.arange(
        -constcent[0] / focal, (W - constcent[0] - 1e-5) / focal, 1 / focal, dtype=torch.float32, device=device
    ).view(1, W)
    directions[1, :, :] = torch.arange(
        constcent[1] / focal, (constcent[1] - H + 1e-5) / focal, -1 / focal, dtype=torch.float32, device=device
    ).view(H, 1)
    directions[2, :, :] = -1

    directions = directions.permute(1, 2, 0)
    return directions


def get_rays_with_directions(directions, c2w):
    """
    Apply directions on pose.

    Args:
        directions(Tensor): ray directions.
        c2w(Tensor): pose.

    Returns:
        Tensor: rays tensor.
        Tensor: ray direction tensor.
    """
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d


def generate_rays(single_pose, H, W, focal=None):
    """
    Generate rays according to pose.

    Args:
        single_pose(list): single pose with focal.
        H(int): height.
        W(int): width.
        focal(float): flocal value.

    Returns:
        Tensor: rays.
    """
    if focal is None:
        focal = single_pose[-1, -1]
    constfocal = focal
    if isinstance(constfocal, torch.Tensor) and len(constfocal.shape) == 0:
        constfocal = constfocal.item()
    directions = get_ray_directions_blender_faster(int(H), int(W), constfocal)
    pose = torch.cuda.FloatTensor(single_pose[:3, :4])
    rays_o, rays_d = get_rays_with_directions(directions, pose)
    return torch.cat([rays_o, rays_d], 1)

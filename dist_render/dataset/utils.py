import json
import os

import cv2
import numpy as np
import torch
from kornia import create_meshgrid


def load_json_drone_data(root_dir, split, image_scale, subfolder, debug=False):
    """
    Load drone dataset.

    Args:
        root_dir(str): root dataset directory path.
        split(str): split mode.
        image_scale(float): image scale ratio.
        subfolder(list): subfolder names.

    Returns:
        dict: meta dataset.
    """
    with open(os.path.join(root_dir, f"transforms_{split}.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    if split == "train":
        with open(os.path.join(root_dir, "transforms_test.json"), "r", encoding="utf-8") as f:
            test_meta = json.load(f)
        meta["frames"] += test_meta["frames"]

    if len(subfolder) > 0:
        print("using data from", subfolder)
        meta["frames"] = [f for f in meta["frames"] if f["file_path"].split("/")[-2] in subfolder]

    pfx = ".JPG" if image_scale == 1 else ".png"
    imgfolder = os.path.join(root_dir, f"images_{image_scale}")
    fnames = [
        frame["file_path"].split("/")[-2] + "/" + frame["file_path"].split("/")[-1].split(".")[0] + pfx
        for frame in meta["frames"]
    ]
    poses = np.stack([np.array(frame["transform_matrix"]) for i, frame in enumerate(meta["frames"])])

    img0 = cv2.imread(os.path.join(imgfolder, fnames[0]), cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
    H, W = img0.shape[:2]
    focal = meta["focal"] * (10 / image_scale)

    if debug:
        fnames = fnames[::50]
        poses = poses[::50]
    return {"poses": poses, "fnames": fnames, "hwf": [H, W, focal], "imgfolder": imgfolder}


def load_json_city_data(
    posefile, root_dir, image_scale, scene_scale=100, valid_paths=None, all_block_render_test=False, debug=False
):
    """
    Load city dataset from root directory.

    Args:
        posefile(str): pose file name.
        root_dir(str): dataset root directory path.
        image_scale(float): image scale ratio.
        scene_scale(float): scane scale ratio.
        valid_paths(Union[None, List[str]]): valid paths.
        debug(bool): True or False.
        all_block_render_test(bool): load dataset of all blocks.

    Returns:
        dict: meta dataset.
    """
    with open(os.path.join(root_dir, posefile), "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert image_scale in [5, 10, 20]
    pfx = ".png"
    imgfolder = os.path.join(root_dir, f"images_{image_scale}")
    fnames = [frame["path"].split("/")[-1].split(".")[0] + pfx for frame in meta.values()]
    if all_block_render_test:
        test_fnames = os.path.join(root_dir, f'{"test_fnames.txt"}')
        assert os.path.exists(test_fnames)
        with open(test_fnames, "r", encoding="utf-8") as f:
            test_fnames = [line.strip("\n").split("/")[-1].split(".")[0] + pfx for line in f]

    if valid_paths is None:
        img_valid = [i for i in range(len(fnames)) if os.path.exists(os.path.join(imgfolder, fnames[i]))]
    else:
        assert len(valid_paths) > 0
        img_valid = [
            i
            for i in range(len(fnames))
            if os.path.exists(os.path.join(imgfolder, fnames[i])) and fnames[i] in valid_paths
        ]

    if all_block_render_test:
        test_img_valid = [i for i in img_valid if fnames[i] in test_fnames]
        img_valid = test_img_valid

    fnames = [os.path.join(imgfolder, fnames[i]) for i in img_valid]

    poses = np.stack([np.array(frame["rot_mat"]) for i, frame in enumerate(meta.values()) if i in img_valid])

    # transform_dict = {k: v for i, (k, v) in enumerate(meta.items()) if i in img_valid}
    if debug:
        fnames = fnames[::10]
        poses = poses[::10]

    poses[:, :3, 3] /= scene_scale
    poses[:, :3, -1] /= image_scale

    img0 = cv2.imread(fnames[0], cv2.IMREAD_UNCHANGED)
    H, W = img0.shape[:2]
    return {"poses": poses, "fnames": fnames, "hw": [H, W], "imgfolder": imgfolder}


def filter_scene(datadir, root_dir, partition, filter_ray):
    """
    filter valid path.

    Args:
        datadir(str): pose file name.
        root_dir(str): dataset root directory path.
        partition(str): partition dataset
        filter_ray(bool): if filter rays.
    Returns:
        list: valid paths.
    """
    partition_json = os.path.join(root_dir, "partition.json")
    with open(partition_json, "r", encoding="utf-8") as fp:
        part_cfg = json.load(fp)[datadir][partition]
    cx, cy, cz = part_cfg["cxyz"]
    dx, dy, dz = part_cfg["dxyz"]

    pad = part_cfg["pad"]
    prefix = f"{(cx-dx/2):.1f}_{(cy-dy/2):.1f}_{(cz-dz/2):.1f}_{(cx+dx/2):.1f}_{(cy+dy/2):.1f}_{(cz+dz/2):.1f}"
    lb = [eval(i) for i in prefix.split("_")[:3]]  # pylint: disable=W0123
    ub = [eval(i) for i in prefix.split("_")[-3:]]  # pylint: disable=W0123
    filter_bbox = torch.tensor([lb, ub])
    scene_bbox = filter_bbox.clone()  # copy value only
    scene_bbox = torch.tensor(
        [
            [scene_bbox[0, 0] - pad, scene_bbox[0, 1] - pad, scene_bbox[0, 2]],
            [scene_bbox[1, 0] + pad, scene_bbox[1, 1] + pad, scene_bbox[1, 2]],
        ]
    )

    if filter_ray:
        filter_json = os.path.join(root_dir, f"filter_{prefix}.json")
        with open(filter_json, "r", encoding="utf-8") as fp:
            filtered_dict = json.load(fp)
    filter_thres = 0.99
    valid_paths = [
        k.split("/")[-1]
        for k, v in filtered_dict["frames"].items()
        if v["ratio"] > filter_thres and v["ground_ratio"] > 0.3
    ]

    return valid_paths


def read_poses_city(datadir, root_dir, image_scale, partition, filter_ray, all_block_render_test=False):
    """
    filter valid path.

    Args:
        datadir(str): pose file name.
        root_dir(str): dataset root directory path.
        image_scale(float): image scale ratio.
        partition(str): partition dataset
        filter_ray(bool): if filter rays.
        all_block_render_test(bool): read all block poses.
    Returns:
        list: valid paths.
    """
    valid_paths = filter_scene(datadir, root_dir, partition, filter_ray)
    meta = load_json_city_data(
        "transform.json",
        root_dir,
        image_scale=image_scale,
        valid_paths=valid_paths,
        all_block_render_test=all_block_render_test,
    )
    poses, _, hw, _ = meta.values()
    return poses, hw


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
    directions = get_ray_directions_blender(int(H), int(W), (focal, focal))
    pose = torch.FloatTensor(single_pose[:3, :4]).cuda()
    rays_o, rays_d = get_rays_with_directions(directions, pose)
    return torch.cat([rays_o, rays_d], 1)

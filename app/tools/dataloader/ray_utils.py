import json
import os
import pickle as pk

import cv2
import numpy as np
import torch
from kornia import create_meshgrid
from PIL import Image
from torch import searchsorted
from tqdm import tqdm


def patchify(img, rays_o, rays_d, H, W, ps):
    """
    Patchify for rgb and rays.

    Args:
        img(torch.Tensor): rgb tensor.
        rays_o(torch.Tensor): ray tensor.
        rays_d(torch.Tensor): ray direction tensor.
        H(int): image height.
        W(int): image weight.
        ps(int): patch size.

    Returns:
        List[Tensor]: list of rgb tensors.
        List[Tensor]: list of rays tensors.
    """
    rays_o = rays_o.view(H, W, 3)
    rays_d = rays_d.view(H, W, 3)
    img = img.view(H, W, 3)
    rays_o_patchify = (
        rays_o.unfold(0, size=ps, step=ps).unfold(1, size=ps, step=ps).unfold(2, 3, 3).reshape(-1, ps * ps, 3)
    )
    rays_d_patchify = (
        rays_d.unfold(0, size=ps, step=ps).unfold(1, size=ps, step=ps).unfold(2, 3, 3).reshape(-1, ps * ps, 3)
    )
    ray_patchify = torch.cat([rays_o_patchify, rays_d_patchify], -1)
    img_patchify = img.unfold(0, size=ps, step=ps).unfold(1, size=ps, step=ps).unfold(2, 3, 3).reshape(-1, ps * ps, 3)

    all_rgbs = list(img_patchify)
    all_rays = list(ray_patchify)
    return all_rgbs, all_rays


def listify_matrix(matrix):
    """
    Transform numpy matrix to list.

    Args:
        matrix(np.ndarray): numpy matrix.

    Returns:
        List: list transformed from numpy matrix.
    """
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def read_Image(f_path, transform):
    """
    Read image from file.

    Args:
        f_path(str): file path.
        transform(torchvision.transforms.ToTensor): transform PIL.Image.Image to torch.Tensor.
    """
    img = Image.open(f_path).convert("RGB")
    img = transform(img)
    img = img.view(3, -1).permute(1, 0)
    return img


def load_json_drone_data(root_dir, split, image_scale, subfolder=None, debug=False):
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
    if subfolder is None:
        subfolder = []
    with open(os.path.join(root_dir, f"transforms_{split}.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    if split == "train":
        with open(os.path.join(root_dir, f'{"transforms_test.json"}'), "r", encoding="utf-8") as f:
            test_meta = json.load(f)
        meta["frames"] += test_meta["frames"]

    if len(subfolder) != 0:
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
    return {
        "poses": poses,
        "fnames": fnames,
        "hwf": [H, W, focal],
        "imgfolder": imgfolder,
    }


def load_json_city_data(
    posefile, root_dir, image_scale, scene_scale=100, valid_paths=None, debug=False, all_block_render_test=False
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

    pfx = ".png"
    imgfolder = os.path.join(root_dir, f"images_{image_scale}")

    if all_block_render_test:
        test_fnames = os.path.join(root_dir, f'{"test_fnames.txt"}')
        assert os.path.exists(test_fnames)
        with open(test_fnames, "r", encoding="utf-8") as f:
            fnames = [line.strip("\n").split("/")[-1].split(".")[0] + pfx for line in f]
    else:
        fnames = [frame["path"].split("/")[-1].split(".")[0] + pfx for frame in meta.values()]

    if valid_paths is None:
        img_valid = [i for i in range(len(fnames)) if os.path.exists(os.path.join(imgfolder, fnames[i]))]
    else:
        assert len(valid_paths) > 0
        img_valid = [
            i
            for i in range(len(fnames))
            if os.path.exists(os.path.join(imgfolder, fnames[i])) and fnames[i] in valid_paths
        ]
    fnames = [os.path.join(imgfolder, fnames[i]) for i in img_valid]
    poses = np.stack([np.array(frame["rot_mat"]) for i, frame in enumerate(meta.values()) if i in img_valid])

    if debug:
        fnames = fnames[::10]
        poses = poses[::10]

    poses[:, :3, 3] /= scene_scale
    poses[:, :3, -1] /= image_scale

    img0 = cv2.imread(fnames[0], cv2.IMREAD_UNCHANGED)
    H, W = img0.shape[:2]
    return {"poses": poses, "fnames": fnames, "hw": [H, W], "imgfolder": imgfolder}


def load_json_render_path(pathfolder, posefile, render_skip=1):
    """
    Load render dataset

    Args:
        pathfolder(str): folder path.
        posefile(str): pose file name.
        render_skip(int): skip size.

    Returns:
        np.ndarray: poses.
    """
    assert posefile.endswith("json")
    with open(os.path.join(pathfolder, posefile), "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta["frames"][::render_skip]
    poses = [frame["transform_matrix"] for frame in frames]
    return np.array(poses)


def load_pk_llff_data(posefile, root_dir, image_scale, pca=True, debug=False):
    """
    Load pk data.

    Args:
        posefile(str): pose file name.
        root_dir(str): root directory path.
        image_scale(float): image scale ratio.
        pca(bool): transform pose by pca.
        debug(bool): True or False.

    Returns:
        dict: meta dataset.
    """
    assert posefile.endswith("pk")
    with open(os.path.join(root_dir, "img_pose.pk"), "rb") as f:
        meta = pk.load(f)
    assert image_scale in [1, 2]
    pfx = ".jpg" if image_scale == 1 else ".png"
    fnames = [fn.split(".")[0] + pfx for fn in list(meta.keys())]
    imgfolder = os.path.join(root_dir, f"images_{image_scale}")
    img0 = cv2.imread(os.path.join(imgfolder, fnames[0]), cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
    H, W = img0.shape[:2]

    poses = np.array(list(meta.values()))
    poses = poses[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
    # llff transform coordinate
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:]], -1)

    hwfs = poses[:, :3, -1:] * (2 / image_scale)
    poses = poses[:, :3, :4]  # (N,3,4)

    # pca
    if pca:
        poses, _ = transform_poses_pca(poses)

    poses = np.concatenate([poses, hwfs], -1)

    # normalize to (-1,1)
    xyz = poses[..., 3]
    scale = max(-np.min(xyz), np.max(xyz))
    poses[..., 3] /= scale

    if debug:
        fnames = fnames[::50]
        poses = poses[::50]
    return {"poses": poses, "fnames": fnames, "hw": [H, W], "imgfolder": imgfolder}


def depth2dist(z_vals, cos_angle):
    """
    Transform z depth to dist.

    Args:
        z_vals(Tensor): depth value.
        cos_angle(Tensor): cos angle value.

    Returns:
        Tensor: transformed z_vals.
    """
    device = z_vals.device
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * cos_angle.unsqueeze(-1)
    return dists


def pad_poses(p):
    """
    Pad pose.

    Args:
        p(np.ndarray): numpy pose.

    Returns:
        np.ndarray: padded poses.
    """
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """
    Unpad pose.

    Args:
        p(np.ndarray): numpy pose.

    Returns:
        np.ndarray: unpadded poses.
    """
    return p[..., :3, :4]


def transform_poses_pca(poses):
    """
    Transform pose by pca.

    Args:
        poses(np.ndarray): numpy poses.

    Returns:
        np.ndarray: poses recentered.
        np.ndarray: transform matrix.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean
    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot
    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform


def trans_t(t):
    return torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]).float()


def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def pose_spherical(theta, phi, radius, zval=None):
    """
    Generate spherical  pose.

    Args:
        theta(float): theta value.
        phi(float): phi value.
        radius(float): radius value.
        zval(float): depth value.

    Returns:
        Tensor: spherical pose.
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    if zval is not None:
        c2w[2, 3] = zval
    return c2w


def get_rays(H, W, K, c2w, camera_mode="normal", offset=0.5):
    """
    Get  rays from pose.

    Args:
        H(int): image height.
        W(int): image width.
        K(Tensor): focal.
        c2w(Tensor): pose.
        camera_mode(str): camera mode, nomal or panorama.
        offset(float): direction offset value.

    Returns:
        Tensor: rays tensor.
        Tensor: rays direction tensor.
    """
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    if camera_mode == "normal":
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    elif camera_mode == "panorama":
        theta = 2 * np.pi * (i + offset) / W
        phi = np.pi * (j + offset) / H
        dirs = torch.stack(
            [
                -torch.sin(phi) * torch.sin(theta),
                torch.cos(phi),
                torch.sin(phi) * torch.cos(theta),
            ],
            -1,
        )
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


# from tensorf code
def get_ray_directions(H, W, focal, center=None):
    """
    Generate ray directions.

    Args:
        H(int): image height.
        W(int): image width.
        focal(tuple): focal value.
        center(Union[List, None]): center point.

    Returns:
        Tensor: ray directions.
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]  # +0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)
    # directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions


def get_ray_directions_blender(H, W, focal, center=None):
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
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]  # +0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -torch.ones_like(i)], -1
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


# others
def filtering_rays(aabb, all_rays, chunk=10240 * 5):
    """
    Filter rays by aabb.

    Args:
        aabb(Tensor): aabb value.
        all_rays(Tensor): rays tensor.
        chunk(int): chunk size.

    Returns:
        List: mask bbox.
        float: filtered ratio.
        List: all points.
    """
    N = torch.tensor(all_rays.shape[:-1]).prod()
    mask_filtered = []
    idx_chunks = torch.split(torch.arange(N), chunk)
    all_pts = []
    for idx_chunk in idx_chunks:
        rays_chunk = all_rays[idx_chunk]
        rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (aabb[1] - rays_o) / vec
        rate_b = (aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1)  # least amount of steps needed to get inside bbox.
        t_max = torch.maximum(rate_a, rate_b).amin(-1)  # least amount of steps needed to get outside bbox.
        mask_inbbox = t_max > t_min
        mask_filtered.append(mask_inbbox.cpu())
        d_z = rays_d[:, -1:]
        o_z = rays_o[:, -1:]
        far = -(o_z / d_z)
        pts = rays_o + rays_d * far
        all_pts.append(pts)
    all_pts = torch.cat(all_pts)
    mask_filtered = torch.cat(mask_filtered)
    ratio = torch.sum(mask_filtered) / N
    return mask_filtered, ratio, all_pts


def prefilter_images(
    root_dir,
    image_scale,
    filter_bbox,
    ground_bbox,
    prefix,
    filter_resize=10,
):
    """
    Filter images.

    Args:
        root_dir(str): root directory path.
        image_scale(float): image scale ratio.
        filter_bbox(Tensor): mask bbox.
        ground_bbox(Tensor): scane bbox.
    """
    filter_resize = 10
    meta = load_json_city_data("transform.json", root_dir, image_scale)
    poses, fnames, hw, _ = meta.values()
    H, W = hw
    H = int(H / filter_resize)
    W = int(W / filter_resize)
    N = len(poses)
    idxs = list(range(N))

    filtered_dict = {"stats": {}, "frames": {}}

    for i in tqdm(idxs):
        image_path = fnames[i]
        focal = poses[i, -1, -1] / filter_resize  # (3,5)
        directions = get_ray_directions_blender(int(H), int(W), (focal, focal))
        pose = torch.FloatTensor(poses[i, :3, :4])
        rays_o, rays_d = get_rays_with_directions(directions, pose)
        _, ratio, pts = filtering_rays(filter_bbox, torch.cat([rays_o, rays_d], 1))
        pts_in_ground = (pts[:, :2] > ground_bbox[0]).sum(-1) + (pts[:, :2] < ground_bbox[1]).sum(-1)
        mask_ground = pts_in_ground == 4
        filtered_dict["frames"][image_path] = {
            "ratio": ratio.item(),
            "ground_ratio": mask_ground.sum().item() / (H * W),
        }

    with open(os.path.join(root_dir, f"filter_{prefix}.json"), "w", encoding="utf-8") as fp:
        json.dump(filtered_dict, fp, indent=4)

    print("file saved to", os.path.join(root_dir, f"filter_{prefix}.json"))


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Sample z values by weights.

    Args:
        bins(Tensor): z value tensors.
        weights(Tensor): generated from density feature.
        det(bool): take uniform samples
        pytest(bool): overwrite u with numpy's fixed random numbers.

    Returns:
        Tensor: sampled bins.
    """
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import (
    get_ray_directions_blender,
    get_rays,
    get_rays_with_directions,
    load_json_render_path,
    pose_spherical,
)


class BaseDataset(Dataset):
    """
    Base dataset
    """

    def __init__(self, split="train", downsample=1.0, is_stack=False, enable_lpips=False, args=None):

        self.N_vis = args.N_vis
        self.datadir = args.datadir
        self.dataroot = args.dataroot
        self.root_dir = os.path.join(args.dataroot, args.datadir)
        self.split = split
        self.is_stack = is_stack
        self.transform = T.ToTensor()
        self.white_bg = args.white_bkgd
        self.camera = args.camera

        self.partition = args.partition
        self.args = args
        self.debug = args.debug
        if self.debug:
            print("dataset for debug mode")
        self.filter_ray = args.filter_ray
        self.patch_size = args.patch_size
        self.lpips = enable_lpips
        self.image_scale = int(downsample)

        # render args
        self.render_nframes = args.render_nframes
        self.render_ncircle = args.render_ncircle
        self.render_downward = args.render_downward
        self.render_px = args.render_px
        self.render_fov = args.render_fov
        self.render_focal = 0.5 * self.render_px / np.tan(0.5 * np.deg2rad(self.render_fov))
        self.render_hwf = [
            int(self.render_px),
            int(1920 / 1080 * self.render_px),
            self.render_focal,
        ]

        self.render_spherical = args.render_spherical
        self.render_spherical_zdiff = args.render_spherical_zdiff
        self.render_spherical_radius = args.render_spherical_radius

        self.render_skip = args.render_skip
        self.render_pathid = args.render_pathid

        self.cxyz = None

        # define scene near/far & bbox
        self.near_far = args.render_near_far if self.split == "path" else args.train_near_far
        if args.lb and args.ub:
            self.scene_bbox = torch.tensor([args.lb, args.ub])
            if split == "path":
                self.render_scene_bbox = torch.tensor([args.render_lb, args.render_ub])
        else:
            pass

        # read meta
        self.poses = []
        self.all_rgbs = []
        self.all_rays = []
        self.all_idxs = []
        self.image_paths = []

        self.render_poses = []
        self.render_rays = []

    def read_meta_path(self):
        """
        Read meta dataset splited by path.
        """
        H, W, focal = self.render_hwf
        self.img_wh = [W, H]
        print("path fov", self.render_fov, "hwf", H, W, self.render_focal)
        directions = get_ray_directions_blender(H, W, (focal, focal))

        if self.render_spherical:
            nframes, radius, zdiff = (
                self.render_nframes // self.render_skip,
                self.render_spherical_radius,
                self.render_spherical_zdiff,
            )
            downward = self.render_downward
            angles = np.linspace(0, 360 * self.render_ncircle, nframes + 1)[:-1]
            radiuss = np.linspace(radius, radius, nframes + 1)  # [:-1]
            poses = torch.stack(
                [pose_spherical(angle, downward, radius) for angle, radius in zip(angles, radiuss)],
                0,
            )
            # recenter pose
            if self.cxyz is not None:
                poses[:, 0, 3] += self.cxyz[0]
                poses[:, 1, 3] += self.cxyz[1]
                if self.partition == "sjt":
                    poses[:, 2, 3] += self.cxyz[2] + 2
                else:
                    poses[:, 2, 3] += self.cxyz[2]

            if zdiff > 0:
                zstep = zdiff / nframes * 2
                for i in range(nframes // 2):
                    poses[i, 2, 3] -= zstep * i

                for i in range(nframes // 2, nframes):
                    poses[i, 2, 3] = poses[i, 2, 3] - zdiff + zstep * (i - nframes // 2)

            print("num of poses", len(poses))
        else:
            poses = load_json_render_path(
                pathfolder=os.path.join(self.root_dir, "trajectories"),
                posefile=f"path{self.render_pathid}.json",
                render_skip=self.render_skip,
            )

            nframes = len(poses)

        idxs = list(range(len(poses)))
        for i in tqdm(idxs, desc=f"Loading data {self.split} ({len(idxs)})"):  # img_list:#
            pose = torch.FloatTensor(poses[i])
            if self.camera == "normal":
                rays_o, rays_d = get_rays_with_directions(directions, pose)
            else:
                K = torch.Tensor([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
                rays_o, rays_d = get_rays(H, W, K, pose, camera_mode=self.camera)

            self.render_poses.append(pose)
            self.render_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.render_poses = torch.stack(self.render_poses)
        self.render_rays = torch.stack(self.render_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        self.all_rays = self.render_rays

    def read_meta(self):
        pass

    def stack_rays(self):
        """
        Stack all rays tensors.
        """
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
        elif self.lpips and self.split == "train":
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
        return sample

import math
import os

import imageio
import numpy as np
import torch
from models import *  # noqa: F401, F403 # pylint: disable=W0611,W0401,W0614
from tools.slurm import get_dp_group
from tools.utils import rgb_lpips, rgb_ssim, visualize_depth_numpy
from torch import distributed as dist
from tqdm.auto import tqdm


def renderer_fn(
    rays,
    gridnerf,
    chunk=4096,
    N_samples=-1,
    white_bg=True,
    is_train=False,
    device="cuda",
):
    """
        The core function of the renderer.

    Args:
        rays (torch.Tensor): Tensor representation of rays.
        gridnerf (torch.nn.Module): GridNeRF model.
        chunk (int): The number of rays processed per iteration.
        N_samples (int): The number of sample points along a ray in total.
        white_bg (bool): Decide whether to render synthetic data on a white bkgd.
        is_train (bool): Decide whether to calculate extra_loss or not.
        device (str): The device on which a rays chunk is or will be allocated.

    Returns:
        dict: Rgp_map and depth_map.
        dict: Metric for loss. Empty when is_train is false.
    """
    all_ret = {}
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        ret, extra_loss = gridnerf(
            rays_chunk,
            is_train=is_train,
            white_bg=white_bg,
            N_samples=N_samples,
        )
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(v, 0) for k, v in all_ret.items()}
    return all_ret, extra_loss


@torch.no_grad()
def evaluation(
    test_dataset,
    gridnerf,
    args,
    renderer,
    savePath=None,
    N_vis=5,
    prtx="",
    N_samples=-1,
    white_bg=False,
    compute_extra_metrics=True,
    device="cuda",
    train_eval=False,
):
    """
        Generate rays from test dataset. Render images based on them and compute psnr.

    Args:
        test_dataset (Dataset): Dataset used for testing.
        gridnerf (torch.nn.Module): GridNeRF model.
        args (tools.configs.ArgsConfig): Evaluation configs.
        renderer (function): Rendering function for computing RGB.
        savePath (str): Path where image, video, etc. information is stored.
        N_vis (int): N images to visualize
        prtx (str): prefix of finename.
        N_samples (int): The number of sample points along a ray in total.
        white_bg (bool): Decide whether to render synthetic data on a white bkgd.
        compute_extra_metrics (bool): Decide whether to compute SSIM and LPIPS metrics.
        device (str): The device on which a tensor is or will be allocated.
        train_eval (bool): Decide whether to calculate extra_loss or not.

    Returns:
        list: A list of PSNR for each image.
    """
    gridnerf.eval()
    if args.DDP:
        train_model = gridnerf.module
    else:
        train_model = gridnerf

    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)

    tqdm._instances.clear()

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

    render_images = len(test_dataset.all_rays[0::img_eval_interval])
    print("test_dataset render images", render_images)

    for idx, samples in enumerate(tqdm(test_dataset.all_rays[0::img_eval_interval])):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        if args.DDP or args.DDP_render:
            if args.model_parallel_and_DDP:
                rank_size = math.ceil(rays.shape[0] / args.num_mp_groups)
                rays_list = rays.split(rank_size)
                rays = rays_list[args.dp_rank]
            else:
                rank_size = math.ceil(rays.shape[0] / args.world_size)
                rays_list = rays.split(rank_size)
                rays = rays_list[args.rank]

        all_ret, _ = renderer(
            rays,
            gridnerf,
            chunk=args.render_batch_size,
            N_samples=N_samples,
            white_bg=white_bg,
            device=device,
            is_train=train_eval,
        )

        rgb_map, depth_map = all_ret["rgb_map"], all_ret["depth_map"]
        rgb_map = rgb_map.clamp(0.0, 1.0)

        if args.DDP or args.DDP_render:
            if args.model_parallel_and_DDP:
                world_size = args.num_mp_groups
                group = get_dp_group()
            else:
                world_size = args.world_size
                group = None
            rgb_map_all = [
                torch.zeros((rays_list[i].shape[0], 3), dtype=torch.float32, device=device) for i in range(world_size)
            ]
            depth_map_all = [
                torch.zeros((rays_list[i].shape[0]), dtype=torch.float32, device=device) for i in range(world_size)
            ]
            dist.all_gather(rgb_map_all, rgb_map, group=group)
            dist.all_gather(depth_map_all, depth_map, group=group)
            rgb_map = torch.cat(rgb_map_all, 0)
            depth_map = torch.cat(depth_map_all, 0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        if len(test_dataset.all_rgbs):
            path = test_dataset.image_paths[idxs[idx]]
            postfix = path.split("/")[-1]

            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", args.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", args.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

            rgb_gt = (gt_rgb.numpy() * 255).astype("uint8")
        else:
            postfix = f"{idx:03d}.png"

        if args.rank == 0:
            print(
                f"{savePath}{prtx}{postfix}",
                depth_map.min(),
                depth_map.max(),
                near_far,
                flush=True,
            )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        if train_model.run_nerf:
            rgb_map_nerf, depth_map_nerf = (
                all_ret["rgb_map_nerf"],
                all_ret["depth_map_nerf"],
            )

            if args.DDP:
                if args.model_parallel_and_DDP:
                    world_size = args.num_mp_groups
                    group = get_dp_group()
                else:
                    world_size = args.world_size
                    group = None
                rgb_map_nerf_all = [
                    torch.zeros((rays_list[i].shape[0], 3), dtype=torch.float32, device=device)
                    for i in range(world_size)
                ]
                depth_map_nerf_all = [
                    torch.zeros((rays_list[i].shape[0]), dtype=torch.float32, device=device) for i in range(world_size)
                ]
                dist.all_gather(rgb_map_nerf_all, rgb_map_nerf, group=group)
                dist.all_gather(depth_map_nerf_all, depth_map_nerf, group=group)
                rgb_map_nerf = torch.cat(rgb_map_nerf_all, 0)
                depth_map_nerf = torch.cat(depth_map_nerf_all, 0)

            rgb_map_nerf, depth_map_nerf = (
                rgb_map_nerf.reshape(H, W, 3).cpu(),
                depth_map_nerf.reshape(H, W).cpu(),
            )
            depth_map_nerf, _ = visualize_depth_numpy(depth_map_nerf.numpy(), near_far)
            if len(test_dataset.all_rgbs):
                loss_nerf = torch.mean((rgb_map_nerf - gt_rgb) ** 2)
                if args.rank == 0:
                    print("psnr", -10.0 * np.log(loss_nerf.item()) / np.log(10.0))
        torch.cuda.empty_cache()
        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)

        if savePath is not None:
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            if len(test_dataset.all_rgbs):
                rgb_map = np.concatenate((rgb_gt, rgb_map), axis=1)

            if train_model.run_nerf:
                rgb_map_nerf = (rgb_map_nerf.numpy() * 255).astype("uint8")
                rgb_map_nerf = np.concatenate((rgb_map_nerf, depth_map_nerf), axis=1)
                rgb_map = np.concatenate((rgb_map, rgb_map_nerf), axis=1)

            if args.rank == 0:
                imageio.imwrite(f"{savePath}/{prtx}{postfix}", rgb_map)

    if args.generate_videos and args.rank == 0:
        imageio.mimwrite(
            f"{savePath}/{prtx}video.mp4",
            np.stack(rgb_maps),
            fps=args.render_fps,
            quality=10,
        )
        imageio.mimwrite(
            f"{savePath}/{prtx}depthvideo.mp4",
            np.stack(depth_maps),
            fps=args.render_fps,
            quality=10,
        )
        imageio.mimwrite(
            f"{savePath}/{prtx}rgb_depthvideo.mp4",
            np.concatenate([np.stack(rgb_maps), np.stack(depth_maps)], 2),
            fps=args.render_fps,
            quality=10,
        )

    if PSNRs and args.rank == 0:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr, ssim, l_a, l_v]))
        else:
            # np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray(PSNRs))

    return PSNRs


@torch.no_grad()
def evaluation_static(
    test_dataset,
    gridnerf,
    args,
    renderer,
    savePath=None,
    prtx="",
    N_samples=-1,
    white_bg=False,
    device="cuda",
):
    """
        Generate images and videos based on a pose trajectory.

    Args:
        test_dataset (Dataset): Dataset used for testing.
        gridnerf (torch.nn.Module): GridNeRF model.
        args (tools.configs.ArgsConfig): Evaluation configs.
        renderer (function): Rendering function for computing RGB.
        savePath (str): Path where image, video, etc. information is stored.
        N_vis (int): N images to visualize
        prtx (str): prefix of finename.
        N_samples (int): The number of sample points along a ray in total.
        white_bg (bool): Decide whether to render synthetic data on a white bkgd.
        device (str): The device on which a tensor is or will be allocated.

    """
    rgb_maps = []
    os.makedirs(savePath, exist_ok=True)
    tqdm._instances.clear()

    for _, samples in enumerate(tqdm(test_dataset.all_rays[:1])):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        if args.DDP:
            if args.model_parallel_and_DDP:
                rank_size = math.ceil(rays.shape[0] / args.num_mp_groups)
                rays_list = rays.split(rank_size)
                rays = rays_list[args.dp_rank]
            else:
                rank_size = math.ceil(rays.shape[0] / args.world_size)
                rays_list = rays.split(rank_size)
                rays = rays_list[args.rank]

        for d_i in range(11500)[::10]:
            dummy_idxs = (torch.zeros_like(samples[:, 0], dtype=torch.long) + d_i).to(device)
            all_ret, _ = renderer(
                rays,
                dummy_idxs,
                gridnerf,
                chunk=args.render_batch_size,
                N_samples=N_samples,
                white_bg=white_bg,
                device=device,
            )

            rgb_map = all_ret["rgb_map"]
            rgb_map = rgb_map.clamp(0.0, 1.0)

            if args.DDP:
                if args.model_parallel_and_DDP:
                    world_size = args.num_mp_groups
                    group = get_dp_group()
                else:
                    world_size = args.world_size
                    group = None
                rgb_map_all = [
                    torch.zeros((rays_list[i].shape[0], 3), dtype=torch.float32, device=device)
                    for i in range(world_size)
                ]
                dist.all_gather(rgb_map_all, rgb_map, group=group)
                rgb_map = torch.cat(rgb_map_all, 0)

            rgb_map = rgb_map.reshape(H, W, 3).cpu()
            postfix = f"{d_i:05d}.png"
            rgb_maps.append(rgb_map)
            if savePath is not None:
                if args.rank == 0 and args.ci_test == 0:
                    imageio.imwrite(f"{savePath}/{prtx}{postfix}", rgb_map)

    if args.generate_videos and args.rank == 0 and args.ci_test == 0:
        imageio.mimwrite(
            f"{savePath}/{prtx}video.mp4",
            np.stack(rgb_maps),
            fps=args.render_fps,
            quality=10,
        )


def create_model(args):
    """
        Create model based on args when rendering.

    Args:
        args (tools.configs.ArgsConfig): Model configs.

    Returns:
        torch.nn.Module: GridNeRF model.
    """

    if args.branch_parallel:
        args.model_name += "BranchParallel"

    if args.ckpt == "auto":
        if args.branch_parallel or args.plane_parallel or args.channel_parallel:
            # normally set "auto" to load sub-ckpt for parallel-trained ckpt
            if args.ckpt_type == "sub":
                args.ckpt = f"{args.logfolder}/{args.expname}-sub{args.rank}.th"
            elif args.branch_parallel and args.ckpt_type == "part":
                raise NotImplementedError
            else:
                raise Exception(
                    "Don't known how to load checkpoints, please check the args.ckpt and args.ckpt_type configs"
                    " setting."
                )
        else:
            args.ckpt = f"{args.logfolder}/{args.expname}.th"

    if args.branch_parallel:
        assert args.ckpt is not None and ".th" in args.ckpt, f"Error, args.ckpt is incorrect: {args.ckpt}."
        # check whether loading the stacked-merged ckpt, since the concat-merged ckpt won't be supported.
        assert "stack" in args.ckpt if "merged" in args.ckpt else True, (
            f"Error, you may load the concated-merged ckpt: {args.ckpt}, which is deprecated and won't be supported."
            " Please load the ckpt with suffix 'merged-stack.th'"
        )

    def kwargs_tensors_to_device(kwargs, device):
        # move the tensors in kwargs to target device
        for key, value in kwargs.items():
            if isinstance(value, dict):
                kwargs_tensors_to_device(value, device)
            elif isinstance(value, torch.Tensor):
                kwargs[key] = value.to(device)

    def rm_ddp_prefix_in_state_dict_if_present(state_dict, prefix=".module"):
        # rm ".module" if exists, since that rendering will never use ddp to wrap modules
        # warning: may cause bug if some of the modules exactly have the name that is composed of 'module'
        keys = sorted(state_dict.keys())
        for k in keys:
            name = k.replace(prefix, "")
            state_dict[name] = state_dict.pop(k)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": args.device, "args": args, "is_train": False})
    kwargs_tensors_to_device(kwargs, args.device)

    gridnerf = eval(args.model_name)(**kwargs)  # pylint: disable=W0123
    rm_ddp_prefix_in_state_dict_if_present(ckpt["state_dict"])
    gridnerf.load(ckpt)

    # reso_cur = gridnerf.gridSize.tolist()
    print("load ckpt from", args.ckpt)

    return gridnerf

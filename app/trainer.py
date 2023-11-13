import datetime
import os
import sys

import lpips
import numpy as np
import torch
import wandb
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision.utils import save_image
from tqdm.auto import tqdm

from app.tools.config_parser import ArgsParser
from app.tools.configs import ArgsConfig
from app.tools.render_utils import evaluation, evaluation_static, renderer_fn
from app.tools.slurm import get_dp_group, init_comm_groups, init_distributed_mode
from app.tools.train_utils import (
    DatasetInfo,
    create_model,
    create_optimizer,
    get_preprocessed_loader,
    prep_dataset,
    prep_sampler,
    prep_testdataset,
    save_optimizer,
)
from app.tools.utils import cal_n_samples, check_args, mse2psnr_npy, n_to_reso, st

renderer = renderer_fn


def init_train_env(cmd=None):
    args_parser = ArgsParser(cmd)
    exp_args = args_parser.get_exp_args()
    model_args = args_parser.get_model_args()
    render_args = args_parser.get_render_args()
    train_args = args_parser.get_train_args()

    args = ArgsConfig([exp_args, model_args, render_args, train_args])

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # setup distributed
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    elif "SLURM_PROCID" in os.environ:
        args.distributed = int(os.environ["SLURM_NTASKS"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  # global rank
    args.model_parallel = bool(args.plane_parallel or args.channel_parallel or args.branch_parallel)
    if args.distributed:
        init_distributed_mode(args)
        print("settings for rank ", args.rank)
        print("rank", args.rank)
        print("world_size", args.world_size)
        print("gpu", args.gpu)
        print("local rank", args.local_rank)
        print("device", args.device)
        print(
            f"Training in distributed mode with multiple processes, 1 GPU per process. Process {args.rank}, total"
            f" {args.world_size}."
        )
        if not args.model_parallel:
            args.DDP = True
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.branch_parallel or args.plane_parallel:
        plane_division = args.plane_division
        args.model_parallel_degree = plane_division[0] * plane_division[1]
    elif args.channel_parallel:
        args.model_parallel_degree = args.channel_parallel_size

    if args.model_parallel_and_DDP:
        args.DDP = True
        args.num_mp_groups = int(args.world_size // args.model_parallel_degree)
        args.dp_rank = args.rank // args.model_parallel_degree
        init_comm_groups(model_parallel_degree=args.model_parallel_degree)
    return args


def train(args):
    if args.tensorboard and args.rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        tb_logfolder = f'./runs/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
        writer = SummaryWriter(log_dir=tb_logfolder)

    datadir = args.datadir.split("/")[-1]

    enable_lpips = False
    enable_distort = False
    # create dataset
    if args.use_preprocessed_data:
        test_dataset = prep_testdataset(args)
        train_dataloader = get_preprocessed_loader(args)
        dataset_info = DatasetInfo(test_dataset.scene_bbox, test_dataset.near_far, test_dataset.white_bg)
    else:
        if args.rank == 0:
            train_dataset, test_dataset = prep_dataset(enable_lpips, args)
        else:
            test_dataset = prep_testdataset(args)
        dataset_info = DatasetInfo(test_dataset.scene_bbox, test_dataset.near_far, test_dataset.white_bg)

    white_bg = dataset_info.white_bg

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"
    args.logfolder = logfolder

    if args.optim_dir is not None:
        optim_dir = args.optim_dir
    else:
        optim_dir = logfolder + "/optim/"

    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(optim_dir, exist_ok=True)

    # create model
    gridnerf, reso_cur = create_model(args, dataset_info)
    gridnerf.train()

    ckpt = args.ckpt

    # only sequential model and channel parallel model will be wrapped by DDP when using DDP training.
    if args.DDP and (not args.model_parallel or args.channel_parallel):
        train_model = gridnerf.module
    else:
        train_model = gridnerf

    nsamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
    if args.rank == 0:
        print(reso_cur, nsamples)

    # save args
    f = os.path.join(logfolder, f'args-{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}.txt')
    if args.rank == 0:
        args.save_config(f)
        if args.config is not None:
            f = os.path.join(logfolder, "config.txt")
            with open(f, "w", encoding="utf-8") as file:
                with open(args.config, "r", encoding="utf-8") as sfile:
                    file.write(sfile.read())

    grad_vars = train_model.get_optparam_groups(args.lr_init, args.lr_basis)

    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    # create optimizer
    optimizer = create_optimizer(grad_vars, args)

    update_alphamask_list = args.update_AlphaMask_list

    # upsaple list
    upsamp_list = args.upsamp_list
    n_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]
    print(f"upsamp_list: {upsamp_list}")
    print(f"n_voxel_list: {n_voxel_list}")

    torch.cuda.empty_cache()
    psnrs, psnrs_nerf, psnrs_test = [], [], [0]

    ortho_reg_weight = args.Ortho_weight
    l1_reg_weight = args.L1_weight_inital
    tv_weight_density, tv_weight_app = args.TV_weight_density, args.TV_weight_app

    if enable_lpips or args.add_lpips > 0:
        lp_loss = lpips.LPIPS(net="vgg").to(args.device)
        ps = args.patch_size

    if not args.use_preprocessed_data and args.rank == 0:
        allrays, allrgbs, allidxs, trainingsampler = prep_sampler(enable_lpips, args, train_dataset)
        print("prepare sampler done", flush=True)

    pbar = tqdm(
        range(args.start_iters, args.n_iters),
        # miniters=args.progress_refresh_rate,
        file=sys.stdout,
    )

    if args.wandb:
        wandb.init(project=f"SH32TEST-{args.datadir}-{args.partition}", name=args.expname)
        wandb.config.update(args)

    if args.render_path:
        folder = f"{logfolder}/imgs_test_path"
        os.makedirs(folder, exist_ok=True)
        gridnerf.eval()
        evaluation_static(
            test_dataset,
            gridnerf,
            args,
            renderer,
            folder,
            N_samples=-1,
            white_bg=white_bg,
            device=args.device,
        )
        return

    for iteration in pbar:
        if args.use_preprocessed_data:
            if iteration % len(train_dataloader) == 0 or iteration == args.start_iters:
                batch_iter = iter(train_dataloader)
            rays, rgbs, idxs = next(batch_iter)
            rays_train, rgb_train, idxs_train = rays.to(args.device), rgbs.to(args.device), idxs.to(args.device)
            rays_train = rays_train.view(-1, 6)
            rgb_train = rgb_train.view(-1, 3)
            idxs_train = idxs_train.view(rays_train.shape[0])
        else:
            if args.rank == 0:
                if args.add_upsample and iteration == args.add_upsample:
                    train_dataset, test_dataset = prep_dataset(enable_lpips, args)
                    allrays, allrgbs, allidxs, trainingsampler = prep_sampler(enable_lpips, args, train_dataset)
                    print("upsample training dataset by x2")

                if args.add_lpips > 0 and iteration == args.add_lpips:
                    enable_lpips = True
                    train_dataset, test_dataset = prep_dataset(enable_lpips, args)
                    allrays, allrgbs, allidxs, trainingsampler = prep_sampler(enable_lpips, args, train_dataset)
                    print("reformat dataset with patch samples")

                ray_idx = trainingsampler.nextids()
                rays_train, rgb_train, idxs_train = (
                    allrays[ray_idx].to(args.device),
                    allrgbs[ray_idx].to(args.device),
                    allidxs[ray_idx].to(args.device),
                )
            else:
                if enable_lpips or (args.add_lpips > 0 and iteration == args.add_lpips):
                    enable_lpips = True
                    rays_train = torch.zeros([ps * ps, 6], dtype=torch.float32, device=args.device)
                    rgb_train = torch.zeros([ps * ps, 3], dtype=torch.float32, device=args.device)
                    idxs_train = torch.zeros([ps * ps], dtype=torch.float32, device=args.device)
                else:
                    rays_train = torch.zeros([args.batch_size, 6], dtype=torch.float32, device=args.device)
                    rgb_train = torch.zeros([args.batch_size, 3], dtype=torch.float32, device=args.device)
                    idxs_train = torch.zeros([args.batch_size], dtype=torch.float32, device=args.device)

            if args.distributed:
                dist.broadcast(rays_train, src=0)
                dist.broadcast(rgb_train, src=0)
                dist.broadcast(idxs_train, src=0)

            if args.DDP:
                if args.model_parallel_and_DDP:
                    num_replicas = args.num_mp_groups
                    dp_rank = args.dp_rank
                else:
                    num_replicas = args.world_size
                    dp_rank = args.rank
                rays_train = torch.chunk(rays_train, num_replicas, dim=0)[dp_rank]
                rgb_train = torch.chunk(rgb_train, num_replicas, dim=0)[dp_rank]
                idxs_train = torch.chunk(idxs_train, num_replicas, dim=0)[dp_rank]

        if args.add_distort > 0 and iteration == args.add_distort:
            enable_distort = True

        if enable_lpips:
            rays_train = rays_train.view(-1, 6)
            rgb_train = rgb_train.view(-1, 3)
            idxs_train = idxs_train.view(rays_train.shape[0])

        if not args.encode_app:
            idxs_train = None

        all_ret, extra_loss = renderer(
            rays_train,
            gridnerf,
            chunk=args.batch_size,
            N_samples=nsamples,
            white_bg=white_bg,
            device=args.device,
            is_train=True,
            idxs=idxs_train,
        )
        rgb_map = all_ret["rgb_map"]
        loss = torch.mean((rgb_map - rgb_train) ** 2)
        total_loss = loss

        # additional rgb supervision
        if "rgb_map1" in all_ret:
            rgb_map1 = all_ret["rgb_map1"]
            loss1 = torch.mean((rgb_map1 - rgb_train) ** 2)
            total_loss += loss1
        if train_model.run_nerf:
            nerf_loss = torch.mean((all_ret["rgb_map_nerf"] - rgb_train) ** 2)
            total_loss = total_loss + nerf_loss

        # regularization loss
        if ortho_reg_weight > 0:
            total_loss += ortho_reg_weight * extra_loss["vector_comp_diffs"]
        if l1_reg_weight > 0:
            total_loss += l1_reg_weight * extra_loss["density_L1"]
        if tv_weight_density > 0:
            tv_weight_density *= lr_factor
            total_loss += extra_loss["TVloss_density"] * tv_weight_density
        if tv_weight_app > 0:
            tv_weight_app *= lr_factor
            total_loss += extra_loss["TVloss_app"] * tv_weight_app

        # lpips loss
        if enable_lpips:
            lpips_w = 0.01  # 0.02
            batch_sample_target_s = torch.reshape(rgb_train, [-1, ps, ps, 3]).permute(0, 3, 1, 2).clamp(0, 1)
            ##
            batch_sample_fake = torch.reshape(rgb_map, [-1, ps, ps, 3]).permute(0, 3, 1, 2).clamp(0, 1)
            lpips_loss = torch.mean(lp_loss.forward(batch_sample_fake * 2 - 1, batch_sample_target_s * 2 - 1))
            total_loss = total_loss + lpips_loss * lpips_w

            if train_model.run_nerf:
                batch_sample_fake_nerf = (
                    torch.reshape(all_ret["rgb_map_nerf"], [-1, ps, ps, 3]).permute(0, 3, 1, 2).clamp(0, 1)
                )
                lpips_loss_nerf = torch.mean(
                    lp_loss.forward(batch_sample_fake_nerf * 2 - 1, batch_sample_target_s * 2 - 1)
                )
                total_loss = total_loss + lpips_loss_nerf * lpips_w

        if enable_distort:
            total_loss += torch.mean(all_ret["distort_loss"])
            if "distort_loss1" in all_ret:
                total_loss += torch.mean(all_ret["distort_loss1"])
            if "distort_loss_nerf" in all_ret:
                total_loss += torch.mean(all_ret["distort_loss_nerf"])

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = torch.mean((rgb_map - rgb_train) ** 2).detach().item()
        psnr = mse2psnr_npy(loss)
        psnrs.append(psnr)

        if enable_lpips:
            lpips_loss = lpips_loss.detach().item()

        if enable_distort:
            distort_loss = torch.mean(all_ret["distort_loss"]).detach().item()

        if train_model.run_nerf:
            nerf_loss = nerf_loss.detach().item()
            nerf_psnr = mse2psnr_npy(nerf_loss)
            psnrs_nerf.append(nerf_psnr)

        if "rgb_map1" in all_ret:
            loss1 = loss1.detach().item()
            psnr1 = mse2psnr_npy(loss1)

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        if iteration % args.progress_refresh_rate == 0:

            string = (
                st.BLUE
                + f"[{datadir}-{args.partition}][{args.expname}]"
                + st.RESET
                + st.YELLOW
                + f" Iter {iteration:06d}:"
                + st.RESET
                + st.RED
                + f" psnr={float(np.mean(psnrs)):.2f}"
                + st.RESET
                + st.GREEN
                + f" test={float(np.mean(psnrs_test)):.2f}"
                + st.RESET
                + f" mse={loss:.3f}"
            )
            if enable_lpips:
                string += f" lpips = {lpips_loss:.2f}"
            if enable_distort:
                string += f" distort = {distort_loss:.10f}"
            if train_model.run_nerf:
                string += f" nerf_psnr = {float(np.mean(psnrs_nerf)):.2f}"
            if "rgb_map1" in all_ret:
                string += f" psnr1 = {psnr1:.2f}"

            if args.rank == 0:
                pbar.set_description(string)
            psnrs = []
            psnrs_nerf = []

        if args.wandb:
            repre_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "Train/Loss": loss,
                    "Train/PSNR": psnr,
                    "Test/PSNR": psnrs_test[-1],
                    "Train/tf_new_lrate": repre_lr,
                },
                step=iteration,
            )
            if enable_lpips:
                wandb.log({"Train/LPIPSLoss": lpips_loss}, step=iteration)
            if train_model.run_nerf:
                wandb.log({"Train/PSNR_nerf": nerf_psnr}, step=iteration)

        if args.tensorboard and args.rank == 0:
            repre_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Train/Loss", loss, iteration)
            writer.add_scalar("Train/PSNR", psnr, iteration)
            writer.add_scalar("Test/PSNR", psnrs_test[-1], iteration)
            writer.add_scalar("Train/tf_new_lrate", repre_lr, iteration)
            if enable_lpips:
                writer.add_scalar("Train/LPIPSLoss", lpips_loss, iteration)
            if train_model.run_nerf:
                writer.add_scalar("Train/PSNR_nerf", nerf_psnr, iteration)

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            prtx = f"{iteration:06d}_"
            gridnerf.eval()
            train_model.train_eval = True
            psnrs_test = evaluation(
                test_dataset,
                gridnerf,
                args,
                renderer,
                f"{logfolder}/imgs_vis/",
                N_vis=args.N_vis,
                prtx=prtx,
                N_samples=nsamples,
                white_bg=white_bg,
                compute_extra_metrics=False,
                train_eval=True,
            )
            if args.DDP:
                den_plane = (
                    train_model.density_plane.module
                    if isinstance(train_model.density_plane, NativeDDP)
                    else train_model.density_plane
                )
                app_plane = (
                    train_model.app_plane.module
                    if isinstance(train_model.app_plane, NativeDDP)
                    else train_model.app_plane
                )
                if args.plane_parallel or args.branch_parallel:
                    model_division = args.plane_division
                    parallel_degree = model_division[0] * model_division[1]
                    if args.rank < parallel_degree:
                        train_model.save(f"{logfolder}/{args.expname}-sub{args.rank}.th")
                        print("render saved to", f"{logfolder}/imgs_vis/")

                        opt_save_path = f"{optim_dir}/{args.expname}_opt-sub{args.rank}.th"
                        print(f"optimizer save to {optim_dir}")
                        save_optimizer(optimizer, opt_save_path)

                        for i in [0]:
                            save_image(
                                den_plane[i].permute(1, 0, 2, 3),
                                f"{logfolder}/den_plane_{i}_sub{args.rank}.png",
                            )
                            save_image(
                                app_plane[i].permute(1, 0, 2, 3),
                                f"{logfolder}/app_plane_{i}_sub{args.rank}.png",
                            )
                else:
                    if args.rank == 0:
                        train_model.save(f"{logfolder}/{args.expname}.th")
                        print("render saved to", f"{logfolder}/imgs_vis/")

                        opt_save_path = f"{optim_dir}/{args.expname}_opt.th"
                        print(f"optimizer save to {optim_dir}")
                        save_optimizer(optimizer, opt_save_path)

                        for i in [0]:
                            save_image(
                                den_plane[i].permute(1, 0, 2, 3),
                                f"{logfolder}/den_plane_{i}.png",
                            )
                            save_image(
                                app_plane[i].permute(1, 0, 2, 3),
                                f"{logfolder}/app_plane_{i}.png",
                            )

            else:
                if args.plane_parallel or args.branch_parallel:
                    train_model.save(f"{logfolder}/{args.expname}-sub{args.rank}.th")
                    print("render saved to", f"{logfolder}/imgs_vis/")

                    opt_save_path = f"{optim_dir}/{args.expname}_opt-sub{args.rank}.th"
                    print(f"optimizer save to {optim_dir}")
                    save_optimizer(optimizer, opt_save_path)

                    for i in [0]:
                        save_image(
                            train_model.density_plane[i].permute(1, 0, 2, 3),
                            f"{logfolder}/den_plane_{i}_sub{args.rank}.png",
                        )
                        save_image(
                            train_model.app_plane[i].permute(1, 0, 2, 3),
                            f"{logfolder}/app_plane_{i}_sub{args.rank}.png",
                        )
                else:
                    if args.rank == 0:
                        train_model.save(f"{logfolder}/{args.expname}.th")
                        print("render saved to", f"{logfolder}/imgs_vis/")

                        opt_save_path = f"{optim_dir}/{args.expname}_opt.th"
                        print(f"optimizer save to {optim_dir}")
                        save_optimizer(optimizer, opt_save_path)

                        for i in [0]:
                            save_image(
                                train_model.density_plane[i].permute(1, 0, 2, 3),
                                f"{logfolder}/den_plane_{i}.png",
                            )
                            save_image(
                                train_model.app_plane[i].permute(1, 0, 2, 3),
                                f"{logfolder}/app_plane_{i}.png",
                            )

            gridnerf.train()

        if not ckpt and not train_model.run_nerf:
            increase_alpha_thresh = 0
            if iteration in update_alphamask_list:
                if reso_cur[0] * reso_cur[1] * reso_cur[2] < args.alpha_grid_reso**3:
                    reso_mask = reso_cur

                train_model.updateAlphaMask(tuple(reso_mask), increase_alpha_thresh)
                if args.progressive_alpha:
                    increase_alpha_thresh += 1

        if iteration in upsamp_list:
            if not ckpt:
                n_voxels = n_voxel_list.pop(0)
            else:
                for it in upsamp_list:
                    if iteration >= it:
                        n_voxels = n_voxel_list.pop(0)
                ckpt = None
            reso_cur = n_to_reso(n_voxels, train_model.aabb)
            nsamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            train_model.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                if args.rank == 0:
                    print(st.CYAN + "reset lr to initial" + st.RESET)
                lr_scale = 1
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            if args.DDP and (not args.model_parallel or args.channel_parallel):
                if args.channel_parallel:
                    ddp_group = get_dp_group()
                else:
                    ddp_group = None
                gridnerf = NativeDDP(
                    gridnerf.module,
                    device_ids=[args.local_rank],
                    process_group=ddp_group,
                    find_unused_parameters=True,
                )
            grad_vars = train_model.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if args.add_nerf > 0 and iteration == args.add_nerf:
            train_model.init_nerf(args)
            train_model.run_nerf = True
            if args.lr_upsample_reset:
                if args.rank == 0:
                    print(st.CYAN + "reset lr to initial" + st.RESET)
                lr_scale = 1
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            if args.DDP and (not args.model_parallel or args.channel_parallel):
                if args.channel_parallel:
                    ddp_group = get_dp_group()
                else:
                    ddp_group = None
                gridnerf = NativeDDP(
                    gridnerf.module,
                    device_ids=[args.local_rank],
                    process_group=ddp_group,
                    find_unused_parameters=True,
                )
            grad_vars = train_model.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            if args.rank == 0:
                print("reload grad_vars")
                print(grad_vars[-1])
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    if args.DDP:
        if args.model_parallel:
            if args.channel_parallel:
                parallel_degree = args.channel_parallel_size
            else:
                model_division = args.plane_division
                parallel_degree = model_division[0] * model_division[1]
            if args.rank < parallel_degree:
                train_model.save(f"{logfolder}/{args.expname}-sub{args.rank}.th")
            if args.distributed:
                dist.barrier()
            if args.rank == 0:
                train_model.merge_ckpts(logfolder)
        else:
            if args.rank == 0:
                train_model.save(f"{logfolder}/{args.expname}.th")
    else:
        if args.model_parallel:
            train_model.save(f"{logfolder}/{args.expname}-sub{args.rank}.th")
            if args.distributed:
                dist.barrier()
            train_model.merge_ckpts(logfolder)
        else:
            if args.rank == 0:
                train_model.save(f"{logfolder}/{args.expname}.th")

    if args.rank == 0:
        opt_save_path = f"{logfolder}/{args.expname}_opt.th"
        print(f"save optimizer to {opt_save_path}")
        save_optimizer(optimizer, opt_save_path)

    folder = f"{logfolder}/imgs_test_all"
    os.makedirs(folder, exist_ok=True)
    gridnerf.eval()
    psnrs_test = evaluation(
        test_dataset,
        gridnerf,
        args,
        renderer,
        folder,
        N_vis=args.N_vis,
        N_samples=-1,
        white_bg=white_bg,
        device=args.device,
        train_eval=True,
    )
    all_psnr = np.mean(psnrs_test)
    print(f"======> {args.expname} test all psnr: {all_psnr} <========================")
    if args.tensorboard and args.rank == 0:
        writer.close()
    return all_psnr


if __name__ == "__main__":
    init_args = init_train_env()
    check_args(init_args)
    train(init_args)

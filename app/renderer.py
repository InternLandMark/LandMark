import datetime
import os

import numpy as np
import torch
from tools.config_parser import ArgsParser
from tools.configs import ArgsConfig
from tools.dataloader import dataset_dict
from tools.render_utils import create_model, evaluation, renderer_fn
from tools.slurm import init_distributed_mode

if __name__ == "__main__":
    args_parser = ArgsParser()
    exp_args = args_parser.get_exp_args()
    model_args = args_parser.get_model_args()
    render_args = args_parser.get_render_args()

    args = ArgsConfig([exp_args, model_args, render_args])

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
    if args.distributed:
        init_distributed_mode(args)
        print("settings for rank ", args.rank)
        print("rank", args.rank)
        print("world_size", args.world_size)
        print("gpu", args.gpu)
        print("local rank", args.local_rank)
        print("device", args.device)
        print(
            f'{"Rendering in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."}'
            % (args.rank, args.world_size)
        )
        args.DDP_render = True

        if args.ckpt_type == "sub":
            # if ckpt_type is sub, distributed environment is not used for ddp rendering
            args.DDP_render = False
    else:
        print("Rendering with a single process on 1 GPUs.")
    assert args.rank >= 0

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        split="test",  # "test",
        downsample=args.downsample_train,
        is_stack=True,
        enable_lpips=False,
        args=args,
    )
    white_bg = test_dataset.white_bg

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"

    args.logfolder = logfolder

    gridnerf = create_model(args)
    renderer = renderer_fn

    folder = f"{logfolder}/imgs_test_all/"
    os.makedirs(folder, exist_ok=True)
    PSNRs_test = evaluation(
        test_dataset,
        gridnerf,
        args,
        renderer,
        folder,
        N_vis=-1,
        N_samples=-1,
        white_bg=white_bg,
        device=args.device,
        compute_extra_metrics=args.compute_extra_metrics,
    )
    all_psnr = np.mean(PSNRs_test)
    print(f"======> {args.expname} test all psnr: {all_psnr} <========================")

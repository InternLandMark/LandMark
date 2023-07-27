import numpy as np
import torch
from models import *  # noqa: F401, F403 # pylint: disable=W0611,W0401,W0614
from tools.utils import n_to_reso
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as DSampler

from .dataloader import dataset_dict
from .dataloader.processeddataset import PreprocessedDataset
from .slurm import get_dp_group, get_mp_group, get_mp_part


class DatasetInfo:
    """Class used to describe dataset information."""

    def __init__(self, aabb, near_far, white_bg):
        self.aabb = aabb
        self.near_far = near_far
        self.white_bg = white_bg


class SimpleSampler:
    """Sampler for randomly sampling rays."""

    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


class DistributedSampler:
    """Sampler for randomly sampling rays in DDP"""

    def __init__(self, total, batch, rank, world_size, seed: int = 0):
        self.total = total
        self.batch = batch
        self.curr = total
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.ids = None

    def nextids(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.curr += self.batch * self.world_size
        if self.curr + self.batch * self.world_size > self.total:
            self.ids = torch.randperm(self.total, generator=g)
            self.curr = 0
            self.epoch += 1
        return self.ids[self.curr + self.rank * self.batch : self.curr + self.rank * self.batch + self.batch]


def create_model(args, dataset_info):
    """
        Create model based on args when training.

    Args:
        args (tools.configs.ArgsConfig): Model configs.
        dataset_info (tools.train_utils.DatasetInfo): Class used to describe dataset information.

    Returns:
        torch.nn.Module: GridNeRF model.
        list: current grid size.
    """

    aabb = dataset_info.aabb.to(args.device)

    use_plane_split = True

    if args.channel_parallel:
        args.model_name += "ChannelParallel"
        use_plane_split = False

    if args.plane_parallel:
        args.model_name += "PlaneParallel"

    if args.branch_parallel:
        args.model_name += "BranchParallel"

    if args.distributed:
        if args.model_parallel_and_DDP:
            args.part = get_mp_part()
        else:
            args.part = args.rank

    if args.ckpt == "auto":
        if args.channel_parallel or args.plane_parallel or args.branch_parallel:
            args.ckpt = f"{args.logfolder}/{args.expname}-sub{args.part}.th"
        else:
            args.ckpt = f"{args.logfolder}/{args.expname}.th"

    if args.model_parallel_and_DDP:
        ddp_group = get_dp_group()
        mp_group = get_mp_group()
    else:
        ddp_group = None
        mp_group = None

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": args.device, "args": args})
        kwargs.update({"group": mp_group})
        gridnerf = eval(args.model_name)(**kwargs)  # pylint: disable=W0123
        gridnerf.load(ckpt)
        reso_cur = gridnerf.gridSize.tolist()
        print("load ckpt from", args.ckpt)
    else:
        reso_cur = n_to_reso(args.N_voxel_init, aabb)
        gridnerf = eval(args.model_name)(  # pylint: disable=W0123
            aabb,
            reso_cur,
            device=args.device,
            density_n_comp=args.n_lamb_sigma,
            appearance_n_comp=args.n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=dataset_info.near_far,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
            use_plane_split=use_plane_split,
            args=args,
            group=mp_group,
        )
    if args.DDP:
        gridnerf = NativeDDP(
            gridnerf, device_ids=[args.local_rank], process_group=ddp_group, find_unused_parameters=True
        )
    return gridnerf, reso_cur


def prep_dataset(enable_lpips, args):
    """Prepare dataset used to train."""

    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        split="train",
        downsample=args.downsample_train,
        is_stack=enable_lpips,
        enable_lpips=enable_lpips,
        args=args,
    )
    test_dataset = dataset(
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
        enable_lpips=enable_lpips,
        args=args,
    )
    return train_dataset, test_dataset


def prep_testdataset(args):
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
        args=args,
    )
    return test_dataset


def get_preprocessed_loader(args):
    conf_path = "~/petreloss.conf"
    data_type = args.processed_data_type
    filefolder = (
        args.preprocessed_dir
        + args.dataset_name
        + "/"
        + args.datadir
        + "/ds"
        + str(args.downsample_train)
        + "_"
        + args.partition
        + "/"
    )
    batch_size = args.batch_size // 8192
    dataset = PreprocessedDataset(filefolder, data_type, conf_path)
    if args.DDP:
        if args.model_parallel_and_DDP:
            sampler = DSampler(
                dataset, num_replicas=args.num_mp_groups, rank=args.dp_rank, shuffle=True, seed=0, drop_last=False
            )
        else:
            sampler = DSampler(
                dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, seed=0, drop_last=False
            )
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader


def prep_sampler(enable_lpips, args, train_dataset):
    """Prepare rays sampler for training"""
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs

    if args.preload:
        print("preload to cuda")
        allrays = allrays.cuda()
        allrgbs = allrgbs.cuda()

    if args.DDP:
        if enable_lpips:
            trainingsampler = DistributedSampler(allrays.shape[0], 1, args.rank, args.world_size)
        else:
            trainingsampler = DistributedSampler(allrays.shape[0], args.batch_size, args.rank, args.world_size)
    else:
        if enable_lpips:
            trainingsampler = SimpleSampler(allrays.shape[0], 1)
        else:
            trainingsampler = SimpleSampler(allrays.shape[0], args.batch_size)
    return allrays, allrgbs, trainingsampler

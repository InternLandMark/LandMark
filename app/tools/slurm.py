import datetime
import os
import subprocess

import torch
import torch.distributed as dist

_MODEL_PARALLEL_GROUP = None
_DP_GROUP = None
_MODEL_PARALLEL_RANKS = []
_MODEL_PARALLEL_PART = None


def setup_for_distributed(is_master):
    """This function disables printing when not in master process"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def ud_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = ud_print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ["LOCAL_SIZE"])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """Set up distributed environment for running. Currently supports single_node/aliyun/slurm only.

    Args:
        args (tools.configs.ArgsConfig): Experimental configs.
    """
    if args.env == "aliyun":
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        CURRENT_RANK = int(os.environ.get("RANK", 99))
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        print("ali_WORLD_SIZE:", WORLD_SIZE)
        print("ali_CURRENT_RANK:", CURRENT_RANK)

        args.dist_backend = "nccl"
        args.dist_url = "env://"
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=WORLD_SIZE,
            rank=CURRENT_RANK,
            timeout=datetime.timedelta(seconds=7200),
        )
        torch.distributed.barrier()
        args.rank = torch.distributed.get_rank()
        setup_for_distributed(args.rank == 0)

        args.local_rank = LOCAL_RANK
        args.gpu = LOCAL_RANK
        args.device = f"cuda:{args.local_rank}"

        args.world_size = torch.distributed.get_world_size()
        args.group = None
        os.environ["LOCAL_SIZE"] = str(torch.cuda.device_count())
        print("args.world_size:", args.world_size)
        torch.cuda.set_device(args.gpu)

        print(f'{"| distributed init (rank {}): {}"}'.format(args.rank, args.dist_url), flush=True)

    elif args.env == "slurm":
        assert "SLURM_PROCID" in os.environ
        print("SLURM PROCID here")
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29510")  # 29500
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
        os.environ["LOCAL_SIZE"] = str(num_gpus)
        args.local_rank = proc_id % num_gpus
        args.dist_url = "env://"
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus  # local rank
        args.device = f"cuda:{args.gpu}"
        args.group = None
        torch.cuda.set_device(args.gpu)
        args.dist_backend = "nccl"
        print(f'{"| distributed init (rank {}): {}"}'.format(args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)
    elif args.env == "single_node":
        if "WORLD_SIZE" in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
            args.local_rank = int(os.environ["LOCAL_RANK"])
            args.device = f"cuda:{args.local_rank}"
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.rank = int(os.environ.get("RANK", 99))
            args.group = None
            args.gpu = int(os.environ["LOCAL_RANK"])
            args.dist_url = "env://"
            os.environ["LOCAL_SIZE"] = str(torch.cuda.device_count())
            args.distributed = True

            torch.cuda.set_device(args.gpu)
            args.dist_backend = "nccl"
            print(f'{"| distributed init (rank {}): {}"}'.format(args.rank, args.dist_url), flush=True)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
            torch.distributed.barrier()
            setup_for_distributed(args.rank == 0)
    else:
        raise Exception(f"Unsupport env {args.env}")


def init_comm_groups(model_parallel_degree=8):
    """Init parallel commuication group

    Args:
        branch_parallel_degree (int): How many ranks in a single branch parallel group.
    """
    global _DP_GROUP
    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_RANKS
    global _MODEL_PARALLEL_PART

    assert dist.is_initialized()
    world_size = dist.get_world_size()
    cur_rank = dist.get_rank()

    assert world_size % model_parallel_degree == 0
    num_mp_groups = int(world_size // model_parallel_degree)
    for i in range(num_mp_groups):
        mp_ranks = range(model_parallel_degree * i, model_parallel_degree * (i + 1))
        # global all_mp_group_ranks
        # all_mp_group_ranks.append(list(tp_ranks))
        # if not use_link:
        mp_group = torch.distributed.new_group(mp_ranks)
        if cur_rank in mp_ranks:
            _MODEL_PARALLEL_GROUP = mp_group
            _MODEL_PARALLEL_RANKS = mp_ranks

    num_dp_groups = model_parallel_degree
    dp_group_size = int(world_size // model_parallel_degree)
    for g in range(num_dp_groups):
        dp_ranks = [g + j * num_dp_groups for j in range(dp_group_size)]
        dp_group = torch.distributed.new_group(dp_ranks)
        print("dp_ranks:", dp_ranks)
        if cur_rank in dp_ranks:
            _DP_GROUP = dp_group

    _MODEL_PARALLEL_PART = cur_rank % model_parallel_degree


def get_dp_group():
    # global _DP_GROUP
    return _DP_GROUP


def get_mp_group():
    # global _MODEL_PARALLEL_GROUP
    return _MODEL_PARALLEL_GROUP


def get_mp_rank0():
    # global _MODEL_PARALLEL_RANKS
    return _MODEL_PARALLEL_RANKS[0]


def get_mp_part():
    # global _MODEL_PARALLEL_PART
    return _MODEL_PARALLEL_PART

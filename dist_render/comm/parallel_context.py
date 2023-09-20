import datetime
import os
from enum import Enum

import torch
from comm.env import EnvSetting
from comm.singleton import SingletonMeta


class ParallelGroup(Enum):
    """
    specified parallel groups used in nerf.
    """

    AllProcesses = "AllProcess"
    ProcessesPerNode = "ProcessesPerNode"
    ProcessesInTenParalGroup = "ProcessesInTenParalGroup"
    ProcessesSameLocalRankBetTenParalGroup = "ProcessesSameLocalRankBetTenParalGroup"
    ProcessesInDataParalGroup = "ProcessesInDataParalGroup"
    ProcessesPerTenParalRank0 = "ProcessesPerTenParalRank0"


class ParallelContext(metaclass=SingletonMeta):
    """
    Parallel context for torch distribution.
    """

    def __init__(self) -> None:
        self._global_rank = 0
        self._world_size = 0
        self._tensor_parallel_size = 1
        self._data_parallel_size = 0
        self._node_cuda_num = 8
        # Build communication groups
        self._groups = {}

    def setup_distributed(self, world_size, rank, backend, dist_url, timeout=1800):
        """
        set torch dist process group env.

        Args:
            world_size(int): global world size.
            rank(int): global rank.
            backend(str): `nccl` default.
            dist_url(str): init method of process group.
            timeout(int): torch dist process group init timeout.
        """
        if world_size is None and rank is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
            torch.distributed.init_process_group(
                backend=backend, init_method=dist_url, timeout=datetime.timedelta(0, timeout)
            )
            torch.cuda.set_device(f"cuda:{torch.distributed.get_rank()}")
        else:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=dist_url,
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(0, timeout),
            )
            torch.distributed.barrier()
            torch.cuda.set_device(f"cuda:{rank%8}")

    def init_distributed(self, world_size=None, rank=None, backend="nccl", dist_url="env://", timeout=1800):
        """
        init torch dist process group env.

        Args:
            world_size(int): global world size.
            rank(int): global rank.
            backend(str): `nccl` default.
            dist_url(str): init method of process group.
            timeout(int): torch dist process group init timeout.
        """
        self.setup_distributed(world_size, rank, backend, dist_url, timeout)
        self._world_size = torch.distributed.get_world_size()
        self._global_rank = torch.distributed.get_rank()

    def register_group(self, group_ranks, parallel_group, local_rank_value=None, register_force=False):
        """
        register parallel group between several process groups.

        Args:
            group_ranks(list): processes group global ranks.
            parallel_group(ParallelGroup): pre-defined parallel groups.
            local_rank_value(int): loal rank value.
            register_force(bool): whether to record group info forcely.
        """
        process_group = torch.distributed.new_group(group_ranks)
        if self.get_rank() in group_ranks:
            group_info = {}
            group_info["local_rank"] = group_ranks.index(self.get_rank())
            group_info["local_rank_value"] = local_rank_value
            group_info["group_world_size"] = len(group_ranks)
            group_info["process_group"] = process_group
            group_info["ranks_in_group"] = group_ranks
            self._groups[parallel_group] = group_info
        elif register_force:
            group_info = {}
            group_info["local_rank"] = None
            group_info["group_world_size"] = len(group_ranks)
            group_info["process_group"] = process_group
            group_info["ranks_in_group"] = group_ranks
            self._groups[parallel_group] = group_info

    def init_groups(self, tensor_parallel_size):
        """
        register all parallel groups used in nerf.

        Args:
            tensor_parallel_size(int): tensor parallel group world size.
        """
        self._tensor_parallel_size = tensor_parallel_size

        # All processes group
        global_ranks = list(range(self.get_world_size()))
        self.register_group(global_ranks, ParallelGroup.AllProcesses)

        # Data parallel processes group
        self._data_parallel_size = self.get_world_size() // self._tensor_parallel_size
        if self._data_parallel_size >= 1:
            global_ranks = [i * self._tensor_parallel_size for i in range(self._data_parallel_size)]
            self.register_group(global_ranks, ParallelGroup.ProcessesInDataParalGroup, register_force=True)

        if self._tensor_parallel_size > 1:
            # Tensor parallel group
            assert self.get_world_size() % self._tensor_parallel_size == 0
            tensor_parallel_group_num = self.get_world_size() // self._tensor_parallel_size
            for i in range(tensor_parallel_group_num):
                global_ranks = [j + i * self._tensor_parallel_size for j in range(self._tensor_parallel_size)]
                self.register_group(global_ranks, ParallelGroup.ProcessesInTenParalGroup)

            # Processes per tensor parallel rank0
            global_ranks = [i * self._tensor_parallel_size for i in range(tensor_parallel_group_num)]
            self.register_group(global_ranks, ParallelGroup.ProcessesPerTenParalRank0)

            # Processes of same tensor parallel local rank on all nodes
            for i in range(self._tensor_parallel_size):
                global_ranks = [i + j * self._tensor_parallel_size for j in range(tensor_parallel_group_num)]
                self.register_group(global_ranks, ParallelGroup.ProcessesSameLocalRankBetTenParalGroup, i)

            # Processes per node
            node_group_num = self.get_world_size() // self._node_cuda_num
            for i in range(node_group_num):
                global_ranks = [j + i * self._node_cuda_num for j in range(self._node_cuda_num)]
                self.register_group(global_ranks, ParallelGroup.ProcessesPerNode)

    def is_group0(self, parallel_group=ParallelGroup.ProcessesInTenParalGroup):
        """
        check whether current process group is in the first group.

        Args:
            parallel_group(ParallelGroup): parallel group defined in advance.
        """
        group0_ranks = list(range(self.get_group_world_size(parallel_group)))
        return self.get_rank() in group0_ranks

    def is_group_rank0(self, parallel_group=ParallelGroup.AllProcesses):
        """
        check whether current process group is the first rank in the given group.
        """
        return self.get_rank() == self.get_group_src_rank(parallel_group=parallel_group)

    def get_group(self, parallel_group=ParallelGroup.AllProcesses):
        """
        get torch process group instance.
        """
        return self._groups[parallel_group]["process_group"]

    def get_group_world_size(self, parallel_group=ParallelGroup.AllProcesses):
        """
        get given group world size.
        """
        return self._groups[parallel_group]["group_world_size"]

    def get_group_src_rank(self, parallel_group=ParallelGroup.AllProcesses):
        """
        get first rank in group
        """
        return self._groups[parallel_group]["ranks_in_group"][0]

    def get_local_rank(self, parallel_group=ParallelGroup.AllProcesses):
        """
        get local rank of the given parallel group.
        """
        return self._groups[parallel_group]["local_rank"]

    def get_rank(self):
        """
        get global rank
        """
        return self._global_rank

    def get_world_size(self):
        """
        get global world size
        """
        return self._world_size

    def get_tensor_parallel_size(self):
        """
        get tensor parallel world size.
        """
        return self._tensor_parallel_size

    def get_data_parallel_size(self):
        """
        get data parallel world size.
        """
        return self._data_parallel_size


def init_parallel_context():
    """
    init parallel context by different ways according to the cluster type.
    """
    ParallelContext().init_distributed(
        world_size=EnvSetting.WORLD_SIZE,
        rank=EnvSetting.RANK,
        backend="nccl",
        dist_url="env://",
        timeout=EnvSetting.NCCL_TIMEOUT,
    )

    # Init groups
    ParallelContext().init_groups(tensor_parallel_size=EnvSetting.TENSOR_PARALLEL_SIZE)

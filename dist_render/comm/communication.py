import torch
from torch import distributed as dist

from dist_render.comm.parallel_context import ParallelContext, ParallelGroup


def scatter(tensor, scatter_list=None, async_op=False, parallel_group=ParallelGroup.AllProcesses):
    """
    custom scatter operation.

    Args:
        tensor(Tensor): Output tensor.
        scatter_list(list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank).
        async_op(bool): Whether this op should be an async op.
        parallel_group(ParallelGroup): Parallel group type we defined in advance.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    src = ParallelContext().get_group_src_rank(parallel_group)
    group = ParallelContext().get_group(parallel_group=parallel_group)
    return dist.scatter(tensor=tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)


def all_gather(tensor, tensor_list=None, async_op=False, parallel_group=ParallelGroup.AllProcesses):
    """
    custom all_gather operation.

    Args:
        tensor(Tensor): Tensor to be broadcast from current process.
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        async_op (bool): Whether this op should be an async op.
        parallel_group(ParallelGroup): Parallel group type we defined in advance.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = ParallelContext().get_group(parallel_group=parallel_group)
    if tensor_list is None:
        group_world_size = ParallelContext().get_group_world_size(parallel_group=parallel_group)
        tensor_list = [torch.empty_like(tensor) for _ in range(group_world_size)]
        res = dist.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)
        return res, tensor_list
    return dist.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)


def all_reduce(tensor, async_op=False, parallel_group=ParallelGroup.AllProcesses):
    """
    custom all_reduce operation.

    Args:
        tensor(Tensor): Tensor to be reduce from current process.
        async_op (bool): Whether this op should be an async op.
        parallel_group(ParallelGroup): Parallel group type we defined in advance.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = ParallelContext().get_group(parallel_group=parallel_group)
    return dist.all_reduce(tensor=tensor, group=group, async_op=async_op)


def gather(tensor, gather_list=None, dst=0, parallel_group=None, async_op=False):
    """
    custom gather operation.

    Args:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        async_op (bool): Whether this op should be an async op.
        parallel_group(ParallelGroup): Parallel group type we defined in advance.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    dst = ParallelContext().get_group_src_rank(parallel_group=parallel_group)
    group = ParallelContext().get_group(parallel_group=parallel_group)
    return dist.gather(tensor, gather_list, dst, group, async_op)


def broadcast(tensor, src=None, parallel_group=None, async_op=False):
    """
    custom broadcast operation.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        async_op (bool): Whether this op should be an async op.
        group (ProcessGroup): The process group to work on. If None,
            the default process group will be used.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    if not src:
        src = ParallelContext().get_group_src_rank(parallel_group=parallel_group)
    group = ParallelContext().get_group(parallel_group=parallel_group)
    return dist.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


def broadcast_object_list(object_list, src=None, parallel_group=None, device=None):
    if not src:
        src = ParallelContext().get_group_src_rank(parallel_group=parallel_group)
    group = ParallelContext().get_group(parallel_group=parallel_group)
    return dist.broadcast_object_list(object_list=object_list, src=src, group=group, device=device)

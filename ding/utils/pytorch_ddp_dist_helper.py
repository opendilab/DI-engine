from typing import Callable, Tuple, List, Any, Union
from easydict import EasyDict

import os
import numpy as np
import torch
import torch.distributed as dist
import datetime

from .default_helper import error_wrapper

# from .slurm_helper import get_master_addr


def get_rank() -> int:
    """
    Overview:
        Get the rank of current process in total world_size
    """
    # return int(os.environ.get('SLURM_PROCID', 0))
    return error_wrapper(dist.get_rank, 0)()


def get_world_size() -> int:
    """
    Overview:
        Get the world_size(total process number in data parallel training)
    """
    # return int(os.environ.get('SLURM_NTASKS', 1))
    return error_wrapper(dist.get_world_size, 1)()


broadcast = dist.broadcast
allgather = dist.all_gather
broadcast_object_list = dist.broadcast_object_list


def allreduce(x: torch.Tensor) -> None:
    """
    Overview:
        All reduce the tensor ``x`` in the world
    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to be reduced
    """

    dist.all_reduce(x)
    x.div_(get_world_size())


def allreduce_with_indicator(grad: torch.Tensor, indicator: torch.Tensor) -> None:
    """
    Overview:
        Custom allreduce: Sum both the gradient and indicator tensors across all processes.
        Then, if at least one process contributed (i.e., the summation of indicator > 0),
        divide the gradient by the summed indicator. This ensures that if only a subset of 
        GPUs contributed a gradient, the averaging is performed based on the actual number
        of contributors rather than the total number of GPUs.
    
    Arguments:
        - grad (torch.Tensor): Local gradient tensor to be reduced.
        - indicator (torch.Tensor): A tensor flag (1 if the gradient is computed, 0 otherwise).
    """
    # Allreduce (sum) the gradient and indicator
    dist.all_reduce(grad)
    dist.all_reduce(indicator)

    # Avoid division by zero. If indicator is close to 0 (extreme case), grad remains zeros.
    if not torch.isclose(indicator, torch.tensor(0.0)):
        grad.div_(indicator.item())


def allreduce_async(name: str, x: torch.Tensor) -> None:
    """
    Overview:
        All reduce the tensor ``x`` in the world asynchronously
    Arguments:
        - name (:obj:`str`): the name of the tensor
        - x (:obj:`torch.Tensor`): the tensor to be reduced
    """

    x.div_(get_world_size())
    dist.all_reduce(x, async_op=True)


def reduce_data(x: Union[int, float, torch.Tensor], dst: int) -> Union[int, float, torch.Tensor]:
    """
    Overview:
        Reduce the tensor ``x`` to the destination process ``dst``
    Arguments:
        - x (:obj:`Union[int, float, torch.Tensor]`): the tensor to be reduced
        - dst (:obj:`int`): the destination process
    """

    if np.isscalar(x):
        x_tensor = torch.as_tensor([x]).cuda()
        dist.reduce(x_tensor, dst)
        return x_tensor.item()
    elif isinstance(x, torch.Tensor):
        dist.reduce(x, dst)
        return x
    else:
        raise TypeError("not supported type: {}".format(type(x)))


def allreduce_data(x: Union[int, float, torch.Tensor], op: str) -> Union[int, float, torch.Tensor]:
    """
    Overview:
        All reduce the tensor ``x`` in the world
    Arguments:
        - x (:obj:`Union[int, float, torch.Tensor]`): the tensor to be reduced
        - op (:obj:`str`): the operation to perform on data, support ``['sum', 'avg']``
    """

    assert op in ['sum', 'avg'], op
    if np.isscalar(x):
        x_tensor = torch.as_tensor([x]).cuda()
        dist.all_reduce(x_tensor)
        if op == 'avg':
            x_tensor.div_(get_world_size())
        return x_tensor.item()
    elif isinstance(x, torch.Tensor):
        dist.all_reduce(x)
        if op == 'avg':
            x.div_(get_world_size())
        return x
    else:
        raise TypeError("not supported type: {}".format(type(x)))


synchronize = torch.cuda.synchronize


def get_group(group_size: int) -> List:
    """
    Overview:
        Get the group segmentation of ``group_size`` each group
    Arguments:
        - group_size (:obj:`int`) the ``group_size``
    """
    rank = get_rank()
    world_size = get_world_size()
    if group_size is None:
        group_size = world_size
    assert (world_size % group_size == 0)
    return simple_group_split(world_size, rank, world_size // group_size)


def dist_mode(func: Callable) -> Callable:
    """
    Overview:
        Wrap the function so that in can init and finalize automatically before each call
    Arguments:
        - func (:obj:`Callable`): the function to be wrapped
    """

    def wrapper(*args, **kwargs):
        dist_init()
        func(*args, **kwargs)
        dist_finalize()

    return wrapper


def dist_init(
        backend: str = 'nccl',
        addr: str = None,
        port: str = None,
        rank: int = None,
        world_size: int = None,
        timeout: datetime.timedelta = datetime.timedelta(seconds=60000)
) -> Tuple[int, int]:
    """
    Overview:
        Initialize the distributed training setting.
    Arguments:
        - backend (:obj:`str`): The backend of the distributed training, supports ``['nccl', 'gloo']``.
        - addr (:obj:`str`): The address of the master node.
        - port (:obj:`str`): The port of the master node.
        - rank (:obj:`int`): The rank of the current process.
        - world_size (:obj:`
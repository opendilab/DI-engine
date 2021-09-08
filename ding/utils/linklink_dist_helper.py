from functools import lru_cache
from typing import Callable, Tuple, List, Any

import numpy as np
import torch

from .default_helper import error_wrapper
from .fake_linklink import FakeLink
from .import_helper import try_import_link


@lru_cache()
def get_link():
    return try_import_link()


@lru_cache()
def is_fake_link():
    return isinstance(get_link(), FakeLink)


def get_rank() -> int:
    r"""
    Overview:
        Get the rank of ``linklink`` model, return 0 if use ``FakeLink``.

    .. note::
        Reference ``import_helper.try_import_link`` and ``linklink.get_rank``.
    """
    if is_fake_link():
        return 0
    return error_wrapper(get_link().get_rank, 0, "[WARNING]: call linklink error, return default_ret.")()


def get_world_size() -> int:
    r"""
    Overview:
        Get the ``world_size`` of ``linklink model``, return 0 if use ``FakeLink``.

    .. note::
        Reference ``import_helper.try_import_link`` and ``linklink.get_world_size``.
    """
    if is_fake_link():
        return 1
    return error_wrapper(get_link().get_world_size, 1, "[WARNING]: call linklink error, return default_ret.")()


def broadcast(value: torch.Tensor, rank: int) -> None:
    r"""
    Overview:
        Use ``linklink.broadcast`` and raise error when using ``FakeLink``
    Arguments:
        - value (:obj:`obj`): the value to board cast
        - rank (:obj:`int`): the rank to broadcast on
    """
    if is_fake_link():
        raise NotImplementedError
    get_link().broadcast(value, rank)


def allreduce(data: torch.Tensor, op: str = 'sum') -> None:
    r"""
    Overview:
        Call ``linklink.allreduce`` on the data
    Arguments:
        - data (:obj:`obj`): the data to reduce
        - op (:obj:`str`): the operation to perform on data, support ``['sum', 'max']``
    """
    link_op_map = {'sum': get_link().allreduceOp_t.Sum, 'max': get_link().allreduceOp_t.Max}
    if op not in link_op_map.keys():
        raise KeyError("not support allreduce op type: {}".format(op))
    else:
        link_op = link_op_map[op]
    if is_fake_link():
        return data
    get_link().allreduce(data, reduce_op=link_op)
    if op == 'sum':
        data.div_(get_world_size())


def allreduce_async(data: torch.Tensor, op: str = 'sum') -> None:
    r"""
    Overview:
        Call ``linklink.allreduce_async`` on the data
    Arguments:
        - data (:obj:`obj`): the data to reduce
        - op (:obj:`str`): the operation to perform on data, support ``['sum', 'max']``
    """
    link_op_map = {'sum': get_link().allreduceOp_t.Sum, 'max': get_link().allreduceOp_t.Max}
    if op not in link_op_map.keys():
        raise KeyError("not support allreduce op type: {}".format(op))
    else:
        link_op = link_op_map[op]
    if is_fake_link():
        return data
    if op == 'sum':
        data.div_(get_world_size())
    get_link().allreduce_async(data, reduce_op=link_op)


def get_group(group_size: int) -> List:
    r"""
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
    r"""
    Overview:
        Wrap the function so that in can init and finalize automatically before each call
    """

    def wrapper(*args, **kwargs):
        dist_init()
        func(*args, **kwargs)
        dist_finalize()

    return wrapper


def dist_init(method: str = 'slurm', device_id: int = 0) -> Tuple[int, int]:
    r"""
    Overview:
        Init the distribution
    Arguments:
        - method (:obj:`str`): Support ``['slurm', 'single_node`]``
        - device_id (:obj:`int`): Default device when using ``single_node`` method
    """
    get_link().initialize()
    world_size = get_link().get_world_size()
    rank = get_link().get_rank()

    if method == 'slurm':
        # proc_id = int(os.environ['SLURM_PROCID'])
        # ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
    elif method == 'single_node':
        torch.cuda.set_device(device_id)

    return rank, world_size


def dist_finalize() -> None:
    r"""
    Overview:
        Finalize ``linklink``, see ``linklink.finalize()``
    """
    get_link().finalize()


class DistContext:

    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        dist_init()

    def __exit__(self, *args, **kwargs) -> Any:
        dist_finalize()


def simple_group_split(world_size: int, rank: int, num_groups: int) -> List:
    r"""
    Overview:
        Split the group according to ``worldsize``, ``rank`` and ``num_groups``

    .. note::
        With faulty input, raise ``array split does not result in an equal division``
    """
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(get_link().new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]


def synchronize():
    get_link().synchronize()

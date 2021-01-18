import os

import numpy as np
import torch

from .default_helper import error_wrapper
from .fake_linklink import FakeLink
from .import_helper import try_import_link

link = try_import_link()

is_fake_link = isinstance(link, FakeLink)


def get_rank():
    r"""
    Overview:
        get the rank of linklink model, return 0 if use FakeLink.
    Notes:
        reference import_helper.try_import_link and linklink.get_rank.
    """
    if is_fake_link:
        return 0
    return error_wrapper(link.get_rank, 0)()


def get_world_size():
    r"""
    Overview:
        get the world_size of linklink model, return 0 if use FakeLink.
    Notes:
        reference import_helper.try_import_link and linklink.get_world_size.
    """
    if is_fake_link:
        return 1
    return error_wrapper(link.get_world_size, 1)()


def broadcast(value, rank_num):
    r"""
    Overview:
        use linklink.broadcast and raise error when using FakeLink
    Arguments:
        - value (:obj:`obj`): the value to board cast
        - rank_num (:obj:`int`): the rank to boardcast on
    """
    if is_fake_link:
        raise NotImplementedError
    link.broadcast(value, rank_num)


def allreduce(data, op='sum'):
    r"""
    Overview:
        call linklink.allreduce on the data
    Arguments:
        - data (:obj:`obj`): the data to reduce
        - op (:obj:`str`): the operation to perform on data, support ['sum', 'max']
    """
    link_op_map = {'sum': link.allreduceOp_t.Sum, 'max': link.allreduceOp_t.Max}
    if op not in link_op_map.keys():
        raise KeyError("not support allreduce op type: {}".format(op))
    else:
        link_op = link_op_map[op]
    if is_fake_link:
        return data
    link.allreduce(data, reduce_op=link_op)
    if op == 'sum':
        data.div_(get_world_size())


def get_group(group_size):
    r"""
    Overview:
        get the group segmentation of group_size each group
    Arguments:
        - group_size (:obj:`int`) the group_size
    """
    rank = get_rank()
    world_size = get_world_size()
    if group_size is None:
        group_size = world_size
    assert (world_size % group_size == 0)
    return simple_group_split(world_size, rank, world_size // group_size)


def distributed_mode(func):
    r"""
    Overview:
        wrap the function so that in can init and finalize automatically before each call
    """

    def wrapper(*args, **kwargs):
        dist_init()
        func(*args, **kwargs)
        dist_finalize()

    return wrapper


def dist_init(method='slurm', device_id=0):
    r"""
    Overview:
        init the distribution
    Arguments:
        - method (:obj:`str`): support ['slurm', 'single_node`]
        - device_id (:obj:`int`): default device when using single_node method
    """
    link.initialize()
    world_size = link.get_world_size()
    rank = link.get_rank()

    if method == 'slurm':
        # proc_id = int(os.environ['SLURM_PROCID'])
        # ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
    elif method == 'single_node':
        torch.cuda.set_device(device_id)

    return rank, world_size


def dist_finalize():
    r"""
    Overview:
        finalize linklink, see linklink.finalize()
    """
    link.finalize()


def simple_group_split(world_size, rank, num_groups):
    r"""
    Overview:
        split the group according to worldsize, rank and num_groups
    """
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]

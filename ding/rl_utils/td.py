import copy
import numpy as np
from collections import namedtuple
from typing import Union, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.hpc_rl import hpc_wrapper
from ding.rl_utils.value_rescale import value_transform, value_inv_transform
from ding.torch_utils import to_tensor

q_1step_td_data = namedtuple('q_1step_td_data', ['q', 'next_q', 'act', 'next_act', 'reward', 'done', 'weight'])


def discount_cumsum(x, gamma: float = 1.0) -> np.ndarray:
    assert abs(gamma - 1.) < 1e-5, "gamma equals to 1.0 in original decision transformer paper"
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def q_1step_td_error(
        data: namedtuple,
        gamma: float,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')  # noqa
) -> torch.Tensor:
    q, next_q, act, next_act, reward, done, weight = data
    assert len(act.shape) == 1, act.shape
    assert len(reward.shape) == 1, reward.shape
    batch_range = torch.arange(act.shape[0])
    if weight is None:
        weight = torch.ones_like(reward)
    q_s_a = q[batch_range, act]
    target_q_s_a = next_q[batch_range, next_act]
    target_q_s_a = gamma * (1 - done) * target_q_s_a + reward
    return (criterion(q_s_a, target_q_s_a.detach()) * weight).mean()


q_v_1step_td_data = namedtuple('q_v_1step_td_data', ['q', 'v', 'act', 'reward', 'done', 'weight'])


def q_v_1step_td_error(
        data: namedtuple, gamma: float, criterion: torch.nn.modules = nn.MSELoss(reduction='none')
) -> torch.Tensor:
    # we will use this function in discrete sac algorithm to calculate td error between q and v value.
    q, v, act, reward, done, weight = data
    if len(act.shape) == 1:
        assert len(reward.shape) == 1, reward.shape
        batch_range = torch.arange(act.shape[0])
        if weight is None:
            weight = torch.ones_like(reward)
        q_s_a = q[batch_range, act]
        target_q_s_a = gamma * (1 - done) * v + reward
    else:
        assert len(reward.shape) == 1, reward.shape
        batch_range = torch.arange(act.shape[0])
        actor_range = torch.arange(act.shape[1])
        batch_actor_range = torch.arange(act.shape[0] * act.shape[1])
        if weight is None:
            weight = torch.ones_like(act)
        temp_q = q.reshape(act.shape[0] * act.shape[1], -1)
        temp_act = act.reshape(act.shape[0] * act.shape[1])
        q_s_a = temp_q[batch_actor_range, temp_act]
        q_s_a = q_s_a.reshape(act.shape[0], act.shape[1])
        target_q_s_a = gamma * (1 - done).unsqueeze(1) * v + reward.unsqueeze(1)
    td_error_per_sample = criterion(q_s_a, target_q_s_a.detach())
    return (td_error_per_sample * weight).mean(), td_error_per_sample


def view_similar(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    size = list(x.shape) + [1 for _ in range(len(target.shape) - len(x.shape))]
    return x.view(*size)


nstep_return_data = namedtuple('nstep_return_data', ['reward', 'next_value', 'done'])


def nstep_return(data: namedtuple, gamma: Union[float, list], nstep: int, value_gamma: Optional[torch.Tensor] = None):
    reward, next_value, done = data
    assert reward.shape[0] == nstep
    device = reward.device

    if isinstance(gamma, float):
        reward_factor = torch.ones(nstep).to(device)
        for i in range(1, nstep):
            reward_factor[i] = gamma * reward_factor[i - 1]
        reward_factor = view_similar(reward_factor, reward)
        return_tmp = reward.mul(reward_factor).sum(0)
        if value_gamma is None:
            return_ = return_tmp + (gamma ** nstep) * next_value * (1 - done)
        else:
            return_ = return_tmp + value_gamma * next_value * (1 - done)

    elif isinstance(gamma, list):
        # if gamma is list, for NGU policy case
        reward_factor = torch.ones([nstep + 1, done.shape[0]]).to(device)
        for i in range(1, nstep + 1):
            reward_factor[i] = torch.stack(gamma, dim=0).to(device) * reward_factor[i - 1]
        reward_factor = view_similar(reward_factor, reward)
        return_tmp = reward.mul(reward_factor[:nstep]).sum(0)
        return_ = return_tmp + reward_factor[nstep] * next_value * (1 - done)
    else:
        raise TypeError("The type of gamma should be float or list")

    return return_


dist_1step_td_data = namedtuple(
    'dist_1step_td_data', ['dist', 'next_dist', 'act', 'next_act', 'reward', 'done', 'weight']
)


def dist_1step_td_error(
        data: namedtuple,
        gamma: float,
        v_min: float,
        v_max: float,
        n_atom: int,
) -> torch.Tensor:
    dist, next_dist, act, next_act, reward, done, weight = data
    device = reward.device
    assert len(reward.shape) == 1, reward.shape
    support = torch.linspace(v_min, v_max, n_atom).to(device)
    delta_z = (v_max - v_min) / (n_atom - 1)

    if len(act.shape) == 1:
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        batch_size = act.shape[0]
        batch_range = torch.arange(batch_size)
        if weight is None:
            weight = torch.ones_like(reward)
        next_dist = next_dist[batch_range, next_act].detach()
    else:
        reward = reward.unsqueeze(-1).repeat(1, act.shape[1])
        done = done.unsqueeze(-1).repeat(1, act.shape[1])

        batch_size = act.shape[0] * act.shape[1]
        batch_range = torch.arange(act.shape[0] * act.shape[1])
        action_dim = dist.shape[2]
        dist = dist.reshape(act.shape[0] * act.shape[1], action_dim, -1)
        reward = reward.reshape(act.shape[0] * act.shape[1], -1)
        done = done.reshape(act.shape[0] * act.shape[1], -1)
        next_dist = next_dist.reshape(act.shape[0] * act.shape[1], action_dim, -1)

        next_act = next_act.reshape(act.shape[0] * act.shape[1])
        next_dist = next_dist[batch_range, next_act].detach()
        next_dist = next_dist.reshape(act.shape[0] * act.shape[1], -1)
        act = act.reshape(act.shape[0] * act.shape[1])
        if weight is None:
            weight = torch.ones_like(reward)
    target_z = reward + (1 - done) * gamma * support
    target_z = target_z.clamp(min=v_min, max=v_max)
    b = (target_z - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    # Fix disappearing probability mass when l = b = u (b is int)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (n_atom - 1)) * (l == u)] += 1

    proj_dist = torch.zeros_like(next_dist)
    offset = torch.linspace(0, (batch_size - 1) * n_atom, batch_size).unsqueeze(1).expand(batch_size,
                                                                                          n_atom).long().to(device)
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    log_p = torch.log(dist[batch_range, act])

    loss = -(log_p * proj_dist * weight).sum(-1).mean()

    return loss


dist_nstep_td_data = namedtuple(
    'dist_1step_td_data', ['dist', 'next_n_dist', 'act', 'next_n_act', 'reward', 'done', 'weight']
)


def shape_fn_dntd(args, kwargs):
    r"""
    Overview:
        Return dntd shape for hpc
    Returns:
        shape: [T, B, N, n_atom]
    """
    if len(args) <= 0:
        tmp = [kwargs['data'].reward.shape[0]]
        tmp.extend(list(kwargs['data'].dist.shape))
    else:
        tmp = [args[0].reward.shape[0]]
        tmp.extend(list(args[0].dist.shape))
    return tmp


@hpc_wrapper(
    shape_fn=shape_fn_dntd,
    namedtuple_data=True,
    include_args=[0, 1, 2, 3],
    include_kwargs=['data', 'gamma', 'v_min', 'v_max']
)
def dist_nstep_td_error(
        data: namedtuple,
        gamma: float,
        v_min: float,
        v_max: float,
        n_atom: int,
        nstep: int = 1,
        value_gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Overview:
        Multistep (1 step or n step) td_error for distributed q-learning based algorithm, support single\
            agent case and multi agent case.
    Arguments:
        - data (:obj:`dist_nstep_td_data`): the input data, dist_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - nstep (:obj:`int`): nstep num, default set to 1
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
    Shapes:
        - data (:obj:`dist_nstep_td_data`): the dist_nstep_td_data containing\
            ['dist', 'next_n_dist', 'act', 'reward', 'done', 'weight']
        - dist (:obj:`torch.FloatTensor`): :math:`(B, N, n_atom)` i.e. [batch_size, action_dim, n_atom]
        - next_n_dist (:obj:`torch.FloatTensor`): :math:`(B, N, n_atom)`
        - act (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_act (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
    """
    dist, next_n_dist, act, next_n_act, reward, done, weight = data
    device = reward.device
    reward_factor = torch.ones(nstep).to(device)
    for i in range(1, nstep):
        reward_factor[i] = gamma * reward_factor[i - 1]
    reward = torch.matmul(reward_factor, reward)
    support = torch.linspace(v_min, v_max, n_atom).to(device)
    delta_z = (v_max - v_min) / (n_atom - 1)
    if len(act.shape) == 1:
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        batch_size = act.shape[0]
        batch_range = torch.arange(batch_size)
        if weight is None:
            weight = torch.ones_like(reward)
        elif isinstance(weight, float):
            weight = torch.tensor(weight)

        next_n_dist = next_n_dist[batch_range, next_n_act].detach()
    else:
        reward = reward.unsqueeze(-1).repeat(1, act.shape[1])
        done = done.unsqueeze(-1).repeat(1, act.shape[1])

        batch_size = act.shape[0] * act.shape[1]
        batch_range = torch.arange(act.shape[0] * act.shape[1])
        action_dim = dist.shape[2]
        dist = dist.reshape(act.shape[0] * act.shape[1], action_dim, -1)
        reward = reward.reshape(act.shape[0] * act.shape[1], -1)
        done = done.reshape(act.shape[0] * act.shape[1], -1)
        next_n_dist = next_n_dist.reshape(act.shape[0] * act.shape[1], action_dim, -1)

        next_n_act = next_n_act.reshape(act.shape[0] * act.shape[1])
        next_n_dist = next_n_dist[batch_range, next_n_act].detach()
        next_n_dist = next_n_dist.reshape(act.shape[0] * act.shape[1], -1)
        act = act.reshape(act.shape[0] * act.shape[1])
        if weight is None:
            weight = torch.ones_like(reward)
        elif isinstance(weight, float):
            weight = torch.tensor(weight)

    if value_gamma is None:
        target_z = reward + (1 - done) * (gamma ** nstep) * support
    elif isinstance(value_gamma, float):
        value_gamma = torch.tensor(value_gamma).unsqueeze(-1)
        target_z = reward + (1 - done) * value_gamma * support
    else:
        value_gamma = value_gamma.unsqueeze(-1)
        target_z = reward + (1 - done) * value_gamma * support
    target_z = target_z.clamp(min=v_min, max=v_max)
    b = (target_z - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    # Fix disappearing probability mass when l = b = u (b is int)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (n_atom - 1)) * (l == u)] += 1

    proj_dist = torch.zeros_like(next_n_dist)
    offset = torch.linspace(0, (batch_size - 1) * n_atom, batch_size).unsqueeze(1).expand(batch_size,
                                                                                          n_atom).long().to(device)
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_n_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_n_dist * (b - l.float())).view(-1))

    assert (dist[batch_range, act] > 0.0).all(), ("dist act", dist[batch_range, act], "dist:", dist)
    log_p = torch.log(dist[batch_range, act])

    if len(weight.shape) == 1:
        weight = weight.unsqueeze(-1)

    td_error_per_sample = -(log_p * proj_dist).sum(-1)

    loss = -(log_p * proj_dist * weight).sum(-1).mean()

    return loss, td_error_per_sample


v_1step_td_data = namedtuple('v_1step_td_data', ['v', 'next_v', 'reward', 'done', 'weight'])


def v_1step_td_error(
        data: namedtuple,
        gamma: float,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')  # noqa
) -> torch.Tensor:
    v, next_v, reward, done, weight = data
    if weight is None:
        weight = torch.ones_like(v)
    if len(v.shape) == len(reward.shape):
        if done is not None:
            target_v = gamma * (1 - done) * next_v + reward
        else:
            target_v = gamma * next_v + reward
    else:
        if done is not None:
            target_v = gamma * (1 - done).unsqueeze(1) * next_v + reward.unsqueeze(1)
        else:
            target_v = gamma * next_v + reward.unsqueeze(1)
    td_error_per_sample = criterion(v, target_v.detach())
    return (td_error_per_sample * weight).mean(), td_error_per_sample


v_nstep_td_data = namedtuple('v_nstep_td_data', ['v', 'next_n_v', 'reward', 'done', 'weight', 'value_gamma'])


def v_nstep_td_error(
        data: namedtuple,
        gamma: float,
        nstep: int = 1,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')  # noqa
) -> torch.Tensor:
    r"""
    Overview:
        Multistep (n step) td_error for distributed value based algorithm
    Arguments:
        - data (:obj:`dist_nstep_td_data`): the input data, v_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - nstep (:obj:`int`): nstep num, default set to 1
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
    Shapes:
        - data (:obj:`dist_nstep_td_data`): the v_nstep_td_data containing\
            ['v', 'next_n_v', 'reward', 'done', 'weight', 'value_gamma']
        - v (:obj:`torch.FloatTensor`): :math:`(B, )` i.e. [batch_size, ]
        - next_v (:obj:`torch.FloatTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
        - value_gamma (:obj:`torch.Tensor`): If the remaining data in the buffer is less than n_step\
            we use value_gamma as the gamma discount value for next_v rather than gamma**n_step
    """
    v, next_n_v, reward, done, weight, value_gamma = data
    if weight is None:
        weight = torch.ones_like(v)
    target_v = nstep_return(nstep_return_data(reward, next_n_v, done), gamma, nstep, value_gamma)
    td_error_per_sample = criterion(v, target_v.detach())
    return (td_error_per_sample * weight).mean(), td_error_per_sample


q_nstep_td_data = namedtuple(
    'q_nstep_td_data', ['q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'weight']
)

dqfd_nstep_td_data = namedtuple(
    'dqfd_nstep_td_data', [
        'q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'done_one_step', 'weight', 'new_n_q_one_step',
        'next_n_action_one_step', 'is_expert'
    ]
)


def shape_fn_qntd(args, kwargs):
    r"""
    Overview:
        Return qntd shape for hpc
    Returns:
        shape: [T, B, N]
    """
    if len(args) <= 0:
        tmp = [kwargs['data'].reward.shape[0]]
        tmp.extend(list(kwargs['data'].q.shape))
    else:
        tmp = [args[0].reward.shape[0]]
        tmp.extend(list(args[0].q.shape))
    return tmp


@hpc_wrapper(shape_fn=shape_fn_qntd, namedtuple_data=True, include_args=[0, 1], include_kwargs=['data', 'gamma'])
def q_nstep_td_error(
        data: namedtuple,
        gamma: Union[float, list],
        nstep: int = 1,
        cum_reward: bool = False,
        value_gamma: Optional[torch.Tensor] = None,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none'),
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error for q-learning based algorithm
    Arguments:
        - data (:obj:`q_nstep_td_data`): the input data, q_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - cum_reward (:obj:`bool`): whether to use cumulative nstep reward, which is figured out when collecting data
        - value_gamma (:obj:`torch.Tensor`): gamma discount value for target q_value
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - nstep (:obj:`int`): nstep num, default set to 1
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
        - td_error_per_sample (:obj:`torch.Tensor`): nstep td error, 1-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
            ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        - td_error_per_sample (:obj:`torch.FloatTensor`): :math:`(B, )`
    """
    q, next_n_q, action, next_n_action, reward, done, weight = data
    if weight is None:
        weight = torch.ones_like(reward)
    if len(action.shape) > 1:  # MARL case
        reward = reward.unsqueeze(-1)
        weight = weight.unsqueeze(-1)
        done = done.unsqueeze(-1)
        if value_gamma is not None:
            value_gamma = value_gamma.unsqueeze(-1)

    q_s_a = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    target_q_s_a = next_n_q.gather(-1, next_n_action.unsqueeze(-1)).squeeze(-1)

    if cum_reward:
        if value_gamma is None:
            target_q_s_a = reward + (gamma ** nstep) * target_q_s_a * (1 - done)
        else:
            target_q_s_a = reward + value_gamma * target_q_s_a * (1 - done)
    else:
        target_q_s_a = nstep_return(nstep_return_data(reward, target_q_s_a, done), gamma, nstep, value_gamma)
    td_error_per_sample = criterion(q_s_a, target_q_s_a.detach())
    return (td_error_per_sample * weight).mean(), td_error_per_sample


def shape_fn_qntd_rescale(args, kwargs):
    r"""
    Overview:
        Return qntd_rescale shape for hpc
    Returns:
        shape: [T, B, N]
    """
    if len(args) <= 0:
        tmp = [kwargs['data'].reward.shape[0]]
        tmp.extend(list(kwargs['data'].q.shape))
    else:
        tmp = [args[0].reward.shape[0]]
        tmp.extend(list(args[0].q.shape))
    return tmp


@hpc_wrapper(
    shape_fn=shape_fn_qntd_rescale, namedtuple_data=True, include_args=[0, 1], include_kwargs=['data', 'gamma']
)
def q_nstep_td_error_with_rescale(
    data: namedtuple,
    gamma: Union[float, list],
    nstep: int = 1,
    value_gamma: Optional[torch.Tensor] = None,
    criterion: torch.nn.modules = nn.MSELoss(reduction='none'),
    trans_fn: Callable = value_transform,
    inv_trans_fn: Callable = value_inv_transform,
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error with value rescaling
    Arguments:
        - data (:obj:`q_nstep_td_data`): the input data, q_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - nstep (:obj:`int`): nstep num, default set to 1
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - trans_fn (:obj:`Callable`): value transfrom function, default to value_transform\
            (refer to rl_utils/value_rescale.py)
        - inv_trans_fn (:obj:`Callable`): value inverse transfrom function, default to value_inv_transform\
            (refer to rl_utils/value_rescale.py)
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
        ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
    """
    q, next_n_q, action, next_n_action, reward, done, weight = data
    assert len(action.shape) == 1, action.shape
    if weight is None:
        weight = torch.ones_like(action)

    batch_range = torch.arange(action.shape[0])
    q_s_a = q[batch_range, action]
    target_q_s_a = next_n_q[batch_range, next_n_action]

    target_q_s_a = inv_trans_fn(target_q_s_a)
    target_q_s_a = nstep_return(nstep_return_data(reward, target_q_s_a, done), gamma, nstep, value_gamma)
    target_q_s_a = trans_fn(target_q_s_a)

    td_error_per_sample = criterion(q_s_a, target_q_s_a.detach())
    return (td_error_per_sample * weight).mean(), td_error_per_sample


def dqfd_nstep_td_error(
        data: namedtuple,
        gamma: float,
        lambda_n_step_td: float,
        lambda_supervised_loss: float,
        margin_function: float,
        lambda_one_step_td: float = 1.,
        nstep: int = 1,
        cum_reward: bool = False,
        value_gamma: Optional[torch.Tensor] = None,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none'),
) -> torch.Tensor:
    """
    Overview:
        Multistep n step td_error + 1 step td_error + supervised margin loss or dqfd
    Arguments:
        - data (:obj:`dqfd_nstep_td_data`): the input data, dqfd_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - cum_reward (:obj:`bool`): whether to use cumulative nstep reward, which is figured out when collecting data
        - value_gamma (:obj:`torch.Tensor`): gamma discount value for target q_value
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - nstep (:obj:`int`): nstep num, default set to 10
    Returns:
        - loss (:obj:`torch.Tensor`): Multistep n step td_error + 1 step td_error + supervised margin loss, 0-dim tensor
        - td_error_per_sample (:obj:`torch.Tensor`): Multistep n step td_error + 1 step td_error\
            + supervised margin loss, 1-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
            ['q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'weight'\
                , 'new_n_q_one_step', 'next_n_action_one_step', 'is_expert']
        - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        - td_error_per_sample (:obj:`torch.FloatTensor`): :math:`(B, )`
        - new_n_q_one_step (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - next_n_action_one_step (:obj:`torch.LongTensor`): :math:`(B, )`
        - is_expert (:obj:`int`) : 0 or 1
    """
    q, next_n_q, action, next_n_action, reward, done, done_one_step, weight, new_n_q_one_step, next_n_action_one_step, \
    is_expert = data  # set is_expert flag(expert 1, agent 0)
    assert len(action.shape) == 1, action.shape
    if weight is None:
        weight = torch.ones_like(action)

    batch_range = torch.arange(action.shape[0])
    q_s_a = q[batch_range, action]
    target_q_s_a = next_n_q[batch_range, next_n_action]
    target_q_s_a_one_step = new_n_q_one_step[batch_range, next_n_action_one_step]

    # calculate n-step TD-loss
    if cum_reward:
        if value_gamma is None:
            target_q_s_a = reward + (gamma ** nstep) * target_q_s_a * (1 - done)
        else:
            target_q_s_a = reward + value_gamma * target_q_s_a * (1 - done)
    else:
        target_q_s_a = nstep_return(nstep_return_data(reward, target_q_s_a, done), gamma, nstep, value_gamma)
    td_error_per_sample = criterion(q_s_a, target_q_s_a.detach())

    # calculate 1-step TD-loss
    nstep = 1
    reward = reward[0].unsqueeze(0)  # get the one-step reward
    value_gamma = None
    if cum_reward:
        if value_gamma is None:
            target_q_s_a_one_step = reward + (gamma ** nstep) * target_q_s_a_one_step * (1 - done_one_step)
        else:
            target_q_s_a_one_step = reward + value_gamma * target_q_s_a_one_step * (1 - done_one_step)
    else:
        target_q_s_a_one_step = nstep_return(
            nstep_return_data(reward, target_q_s_a_one_step, done_one_step), gamma, nstep, value_gamma
        )
    td_error_one_step_per_sample = criterion(q_s_a, target_q_s_a_one_step.detach())
    device = q_s_a.device
    device_cpu = torch.device('cpu')
    # calculate the supervised loss
    l = margin_function * torch.ones_like(q).to(device_cpu)  # q shape (B, A), action shape (B, )
    l.scatter_(1, torch.LongTensor(action.unsqueeze(1).to(device_cpu)), torch.zeros_like(q, device=device_cpu))
    # along the first dimension. for the index of the action, fill the corresponding position in l with 0
    JE = is_expert * (torch.max(q + l.to(device), dim=1)[0] - q_s_a)

    return (
        (
            (
                lambda_n_step_td * td_error_per_sample + lambda_one_step_td * td_error_one_step_per_sample +
                lambda_supervised_loss * JE
            ) * weight
        ).mean(), lambda_n_step_td * td_error_per_sample.abs() +
        lambda_one_step_td * td_error_one_step_per_sample.abs() + lambda_supervised_loss * JE.abs(),
        (td_error_per_sample.mean(), td_error_one_step_per_sample.mean(), JE.mean())
    )


def dqfd_nstep_td_error_with_rescale(
    data: namedtuple,
    gamma: float,
    lambda_n_step_td: float,
    lambda_supervised_loss: float,
    lambda_one_step_td: float,
    margin_function: float,
    nstep: int = 1,
    cum_reward: bool = False,
    value_gamma: Optional[torch.Tensor] = None,
    criterion: torch.nn.modules = nn.MSELoss(reduction='none'),
    trans_fn: Callable = value_transform,
    inv_trans_fn: Callable = value_inv_transform,
) -> torch.Tensor:
    """
    Overview:
        Multistep n step td_error + 1 step td_error + supervised margin loss or dqfd
    Arguments:
        - data (:obj:`dqfd_nstep_td_data`): the input data, dqfd_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - cum_reward (:obj:`bool`): whether to use cumulative nstep reward, which is figured out when collecting data
        - value_gamma (:obj:`torch.Tensor`): gamma discount value for target q_value
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - nstep (:obj:`int`): nstep num, default set to 10
    Returns:
        - loss (:obj:`torch.Tensor`): Multistep n step td_error + 1 step td_error + supervised margin loss, 0-dim tensor
        - td_error_per_sample (:obj:`torch.Tensor`): Multistep n step td_error + 1 step td_error\
            + supervised margin loss, 1-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
            ['q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'weight'\
                , 'new_n_q_one_step', 'next_n_action_one_step', 'is_expert']
        - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        - td_error_per_sample (:obj:`torch.FloatTensor`): :math:`(B, )`
        - new_n_q_one_step (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - next_n_action_one_step (:obj:`torch.LongTensor`): :math:`(B, )`
        - is_expert (:obj:`int`) : 0 or 1
    """
    q, next_n_q, action, next_n_action, reward, done, done_one_step, weight, new_n_q_one_step, next_n_action_one_step, \
    is_expert = data  # set is_expert flag(expert 1, agent 0)
    assert len(action.shape) == 1, action.shape
    if weight is None:
        weight = torch.ones_like(action)

    batch_range = torch.arange(action.shape[0])
    q_s_a = q[batch_range, action]

    target_q_s_a = next_n_q[batch_range, next_n_action]
    target_q_s_a = inv_trans_fn(target_q_s_a)  # rescale

    target_q_s_a_one_step = new_n_q_one_step[batch_range, next_n_action_one_step]
    target_q_s_a_one_step = inv_trans_fn(target_q_s_a_one_step)  # rescale

    # calculate n-step TD-loss
    if cum_reward:
        if value_gamma is None:
            target_q_s_a = reward + (gamma ** nstep) * target_q_s_a * (1 - done)
        else:
            target_q_s_a = reward + value_gamma * target_q_s_a * (1 - done)
    else:
        # to use value_gamma in  n-step TD-loss
        target_q_s_a = nstep_return(nstep_return_data(reward, target_q_s_a, done), gamma, nstep, value_gamma)

    target_q_s_a = trans_fn(target_q_s_a)  # rescale
    td_error_per_sample = criterion(q_s_a, target_q_s_a.detach())

    # calculate 1-step TD-loss
    nstep = 1
    reward = reward[0].unsqueeze(0)  # get the one-step reward
    value_gamma = None  # This is very important, to use gamma in  1-step TD-loss
    if cum_reward:
        if value_gamma is None:
            target_q_s_a_one_step = reward + (gamma ** nstep) * target_q_s_a_one_step * (1 - done_one_step)
        else:
            target_q_s_a_one_step = reward + value_gamma * target_q_s_a_one_step * (1 - done_one_step)
    else:
        target_q_s_a_one_step = nstep_return(
            nstep_return_data(reward, target_q_s_a_one_step, done_one_step), gamma, nstep, value_gamma
        )

    target_q_s_a_one_step = trans_fn(target_q_s_a_one_step)  # rescale
    td_error_one_step_per_sample = criterion(q_s_a, target_q_s_a_one_step.detach())
    device = q_s_a.device
    device_cpu = torch.device('cpu')
    # calculate the supervised loss
    l = margin_function * torch.ones_like(q).to(device_cpu)  # q shape (B, A), action shape (B, )
    l.scatter_(1, torch.LongTensor(action.unsqueeze(1).to(device_cpu)), torch.zeros_like(q, device=device_cpu))
    # along the first dimension. for the index of the action, fill the corresponding position in l with 0
    JE = is_expert * (torch.max(q + l.to(device), dim=1)[0] - q_s_a)

    return (
        (
            (
                lambda_n_step_td * td_error_per_sample + lambda_one_step_td * td_error_one_step_per_sample +
                lambda_supervised_loss * JE
            ) * weight
        ).mean(), lambda_n_step_td * td_error_per_sample.abs() +
        lambda_one_step_td * td_error_one_step_per_sample.abs() + lambda_supervised_loss * JE.abs(),
        (td_error_per_sample.mean(), td_error_one_step_per_sample.mean(), JE.mean())
    )


qrdqn_nstep_td_data = namedtuple(
    'qrdqn_nstep_td_data', ['q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'tau', 'weight']
)


def qrdqn_nstep_td_error(
        data: namedtuple,
        gamma: float,
        nstep: int = 1,
        value_gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error with in QRDQN
    Arguments:
        - data (:obj:`iqn_nstep_td_data`): the input data, iqn_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - nstep (:obj:`int`): nstep num, default set to 1
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
        ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(tau, B, N)` i.e. [tau x batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(tau', B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
    """
    q, next_n_q, action, next_n_action, reward, done, tau, weight = data

    assert len(action.shape) == 1, action.shape
    assert len(next_n_action.shape) == 1, next_n_action.shape
    assert len(done.shape) == 1, done.shape
    assert len(q.shape) == 3, q.shape
    assert len(next_n_q.shape) == 3, next_n_q.shape
    assert len(reward.shape) == 2, reward.shape

    if weight is None:
        weight = torch.ones_like(action)

    batch_range = torch.arange(action.shape[0])

    # shape: batch_size x num x 1
    q_s_a = q[batch_range, action, :].unsqueeze(2)
    # shape: batch_size x 1 x num
    target_q_s_a = next_n_q[batch_range, next_n_action, :].unsqueeze(1)

    assert reward.shape[0] == nstep
    reward_factor = torch.ones(nstep).to(reward)
    for i in range(1, nstep):
        reward_factor[i] = gamma * reward_factor[i - 1]
    # shape: batch_size
    reward = torch.matmul(reward_factor, reward)
    # shape: batch_size x 1 x num
    if value_gamma is None:
        target_q_s_a = reward.unsqueeze(-1).unsqueeze(-1) + (gamma ** nstep
                                                             ) * target_q_s_a * (1 - done).unsqueeze(-1).unsqueeze(-1)
    else:
        target_q_s_a = reward.unsqueeze(-1).unsqueeze(
            -1
        ) + value_gamma.unsqueeze(-1).unsqueeze(-1) * target_q_s_a * (1 - done).unsqueeze(-1).unsqueeze(-1)

    # shape: batch_size x num x num
    u = F.smooth_l1_loss(target_q_s_a, q_s_a, reduction="none")
    # shape: batch_size
    loss = (u * (tau - (target_q_s_a - q_s_a).detach().le(0.).float()).abs()).sum(-1).mean(1)

    return (loss * weight).mean(), loss


def q_nstep_sql_td_error(
        data: namedtuple,
        gamma: float,
        alpha: float,
        nstep: int = 1,
        cum_reward: bool = False,
        value_gamma: Optional[torch.Tensor] = None,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none'),
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error for q-learning based algorithm
    Arguments:
        - data (:obj:`q_nstep_td_data`): the input data, q_nstep_sql_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - Alpha (:obj:｀float`): A parameter to weight entropy term in a policy equation
        - cum_reward (:obj:`bool`): whether to use cumulative nstep reward, which is figured out when collecting data
        - value_gamma (:obj:`torch.Tensor`): gamma discount value for target soft_q_value
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - nstep (:obj:`int`): nstep num, default set to 1
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
        - td_error_per_sample (:obj:`torch.Tensor`): nstep td error, 1-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
            ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        - td_error_per_sample (:obj:`torch.FloatTensor`): :math:`(B, )`
    """
    q, next_n_q, action, next_n_action, reward, done, weight = data
    assert len(action.shape) == 1, action.shape
    if weight is None:
        weight = torch.ones_like(action)

    batch_range = torch.arange(action.shape[0])
    q_s_a = q[batch_range, action]
    # target_q_s_a = next_n_q[batch_range, next_n_action]
    target_v = alpha * torch.logsumexp(
        next_n_q / alpha, 1
    )  # target_v = alpha * torch.log(torch.sum(torch.exp(next_n_q / alpha), 1))
    target_v[target_v == float("Inf")] = 20
    target_v[target_v == float("-Inf")] = -20
    # For an appropriate hyper-parameter alpha, these hardcodes can be removed.
    # However, algorithms may face the danger of explosion for other alphas.
    # The hardcodes above are to prevent this situation from happening
    record_target_v = copy.deepcopy(target_v)
    # print(target_v)
    if cum_reward:
        if value_gamma is None:
            target_v = reward + (gamma ** nstep) * target_v * (1 - done)
        else:
            target_v = reward + value_gamma * target_v * (1 - done)
    else:
        target_v = nstep_return(nstep_return_data(reward, target_v, done), gamma, nstep, value_gamma)
    td_error_per_sample = criterion(q_s_a, target_v.detach())
    return (td_error_per_sample * weight).mean(), td_error_per_sample, record_target_v


iqn_nstep_td_data = namedtuple(
    'iqn_nstep_td_data', ['q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'replay_quantiles', 'weight']
)


def iqn_nstep_td_error(
        data: namedtuple,
        gamma: float,
        nstep: int = 1,
        kappa: float = 1.0,
        value_gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error with in IQN, \
            referenced paper Implicit Quantile Networks for Distributional Reinforcement Learning \
            <https://arxiv.org/pdf/1806.06923.pdf>
    Arguments:
        - data (:obj:`iqn_nstep_td_data`): the input data, iqn_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - nstep (:obj:`int`): nstep num, default set to 1
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - beta_function (:obj:`Callable`): the risk function
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
        ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(tau, B, N)` i.e. [tau x batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(tau', B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
    """
    q, next_n_q, action, next_n_action, reward, done, replay_quantiles, weight = data

    assert len(action.shape) == 1, action.shape
    assert len(next_n_action.shape) == 1, next_n_action.shape
    assert len(done.shape) == 1, done.shape
    assert len(q.shape) == 3, q.shape
    assert len(next_n_q.shape) == 3, next_n_q.shape
    assert len(reward.shape) == 2, reward.shape

    if weight is None:
        weight = torch.ones_like(action)

    batch_size = done.shape[0]
    tau = q.shape[0]
    tau_prime = next_n_q.shape[0]

    action = action.repeat([tau, 1]).unsqueeze(-1)
    next_n_action = next_n_action.repeat([tau_prime, 1]).unsqueeze(-1)

    # shape: batch_size x tau x a
    q_s_a = torch.gather(q, -1, action).permute([1, 0, 2])
    # shape: batch_size x tau_prim x 1
    target_q_s_a = torch.gather(next_n_q, -1, next_n_action).permute([1, 0, 2])

    assert reward.shape[0] == nstep
    device = torch.device("cuda" if reward.is_cuda else "cpu")
    reward_factor = torch.ones(nstep).to(device)
    for i in range(1, nstep):
        reward_factor[i] = gamma * reward_factor[i - 1]
    reward = torch.matmul(reward_factor, reward)
    if value_gamma is None:
        target_q_s_a = reward.unsqueeze(-1) + (gamma ** nstep) * target_q_s_a.squeeze(-1) * (1 - done).unsqueeze(-1)
    else:
        target_q_s_a = reward.unsqueeze(-1) + value_gamma.unsqueeze(-1) * target_q_s_a.squeeze(-1) * (1 - done
                                                                                                      ).unsqueeze(-1)
    target_q_s_a = target_q_s_a.unsqueeze(-1)

    # shape: batch_size x tau' x tau x 1.
    bellman_errors = (target_q_s_a[:, :, None, :] - q_s_a[:, None, :, :])

    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    huber_loss = torch.where(
        bellman_errors.abs() <= kappa, 0.5 * bellman_errors ** 2, kappa * (bellman_errors.abs() - 0.5 * kappa)
    )

    # Reshape replay_quantiles to batch_size x num_tau_samples x 1
    replay_quantiles = replay_quantiles.reshape([tau, batch_size, 1]).permute([1, 0, 2])

    # shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    replay_quantiles = replay_quantiles[:, None, :, :].repeat([1, tau_prime, 1, 1])

    # shape: batch_size x tau_prime x tau x 1.
    quantile_huber_loss = (torch.abs(replay_quantiles - ((bellman_errors < 0).float()).detach()) * huber_loss) / kappa

    # shape: batch_size
    loss = quantile_huber_loss.sum(dim=2).mean(dim=1)[:, 0]

    return (loss * weight).mean(), loss


fqf_nstep_td_data = namedtuple(
    'fqf_nstep_td_data', ['q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'quantiles_hats', 'weight']
)


def fqf_nstep_td_error(
        data: namedtuple,
        gamma: float,
        nstep: int = 1,
        kappa: float = 1.0,
        value_gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error with in FQF, \
            referenced paper Fully Parameterized Quantile Function for Distributional Reinforcement Learning \
            <https://arxiv.org/pdf/1911.02140.pdf>
    Arguments:
        - data (:obj:`fqf_nstep_td_data`): the input data, fqf_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - nstep (:obj:`int`): nstep num, default set to 1
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - beta_function (:obj:`Callable`): the risk function
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
        ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(B, tau, N)` i.e. [batch_size, tau, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, tau', N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        - quantiles_hats (:obj:`torch.FloatTensor`): :math:`(B, tau)`
    """
    q, next_n_q, action, next_n_action, reward, done, quantiles_hats, weight = data

    assert len(action.shape) == 1, action.shape
    assert len(next_n_action.shape) == 1, next_n_action.shape
    assert len(done.shape) == 1, done.shape
    assert len(q.shape) == 3, q.shape
    assert len(next_n_q.shape) == 3, next_n_q.shape
    assert len(reward.shape) == 2, reward.shape

    if weight is None:
        weight = torch.ones_like(action)

    batch_size = done.shape[0]
    tau = q.shape[1]
    tau_prime = next_n_q.shape[1]

    # shape: batch_size x tau x 1
    q_s_a = evaluate_quantile_at_action(q, action)
    # shape: batch_size x tau_prime x 1
    target_q_s_a = evaluate_quantile_at_action(next_n_q, next_n_action)

    assert reward.shape[0] == nstep
    reward_factor = torch.ones(nstep).to(reward.device)
    for i in range(1, nstep):
        reward_factor[i] = gamma * reward_factor[i - 1]
    reward = torch.matmul(reward_factor, reward)  # [batch_size]
    if value_gamma is None:
        target_q_s_a = reward.unsqueeze(-1) + (gamma ** nstep) * target_q_s_a.squeeze(-1) * (1 - done).unsqueeze(-1)
    else:
        target_q_s_a = reward.unsqueeze(-1) + value_gamma.unsqueeze(-1) * target_q_s_a.squeeze(-1) * (1 - done
                                                                                                      ).unsqueeze(-1)
    target_q_s_a = target_q_s_a.unsqueeze(-1)

    # shape: batch_size x tau' x tau x 1.
    bellman_errors = (target_q_s_a.unsqueeze(2) - q_s_a.unsqueeze(1))

    # shape: batch_size x tau' x tau x 1
    huber_loss = F.smooth_l1_loss(target_q_s_a.unsqueeze(2), q_s_a.unsqueeze(1), reduction="none")

    # shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantiles_hats = quantiles_hats[:, None, :, None].repeat([1, tau_prime, 1, 1])

    # shape: batch_size x tau_prime x tau x 1.
    quantile_huber_loss = (torch.abs(quantiles_hats - ((bellman_errors < 0).float()).detach()) * huber_loss) / kappa

    # shape: batch_size
    loss = quantile_huber_loss.sum(dim=2).mean(dim=1)[:, 0]

    return (loss * weight).mean(), loss


def evaluate_quantile_at_action(q_s, actions):
    assert q_s.shape[0] == actions.shape[0]

    batch_size, num_quantiles = q_s.shape[:2]

    # Expand actions into (batch_size, num_quantiles, 1).
    action_index = actions[:, None, None].expand(batch_size, num_quantiles, 1)

    # Calculate quantile values at specified actions.
    q_s_a = q_s.gather(dim=2, index=action_index)

    return q_s_a


def fqf_calculate_fraction_loss(q_tau_i, q_value, quantiles, actions):
    """
    Shapes:
       - q_tau_i (:obj:`torch.FloatTensor`) :math:`(batch_size, num_quantiles-1, action_dim)`
       - q_value (:obj:`torch.FloatTensor`) :math:`(batch_size, num_quantiles, action_dim)`
       - quantiles (:obj:`torch.FloatTensor`) :math:`(batch_size, num_quantiles+1)`
       - actions (:obj:`torch.LongTensor`) :math:`(batch_size, )`
    """
    assert q_value.requires_grad

    batch_size = q_value.shape[0]
    num_quantiles = q_value.shape[1]

    with torch.no_grad():
        sa_quantiles = evaluate_quantile_at_action(q_tau_i, actions)
        assert sa_quantiles.shape == (batch_size, num_quantiles - 1, 1)
        q_s_a_hats = evaluate_quantile_at_action(q_value, actions)  # [batch_size, num_quantiles, 1]
        assert q_s_a_hats.shape == (batch_size, num_quantiles, 1)
        assert not q_s_a_hats.requires_grad

    # NOTE: Proposition 1 in the paper requires F^{-1} is non-decreasing.
    # I relax this requirements and calculate gradients of quantiles even when
    # F^{-1} is not non-decreasing.

    values_1 = sa_quantiles - q_s_a_hats[:, :-1]
    signs_1 = sa_quantiles > torch.cat([q_s_a_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
    assert values_1.shape == signs_1.shape

    values_2 = sa_quantiles - q_s_a_hats[:, 1:]
    signs_2 = sa_quantiles < torch.cat([sa_quantiles[:, 1:], q_s_a_hats[:, -1:]], dim=1)
    assert values_2.shape == signs_2.shape

    gradient_of_taus = (torch.where(signs_1, values_1, -values_1) +
                        torch.where(signs_2, values_2, -values_2)).view(batch_size, num_quantiles - 1)
    assert not gradient_of_taus.requires_grad
    assert gradient_of_taus.shape == quantiles[:, 1:-1].shape

    # Gradients of the network parameters and corresponding loss
    # are calculated using chain rule.
    fraction_loss = (gradient_of_taus * quantiles[:, 1:-1]).sum(dim=1).mean()

    return fraction_loss


td_lambda_data = namedtuple('td_lambda_data', ['value', 'reward', 'weight'])


def shape_fn_td_lambda(args, kwargs):
    r"""
    Overview:
        Return td_lambda shape for hpc
    Returns:
        shape: [T, B]
    """
    if len(args) <= 0:
        tmp = kwargs['data'].reward.shape[0]
    else:
        tmp = args[0].reward.shape
    return tmp


@hpc_wrapper(
    shape_fn=shape_fn_td_lambda,
    namedtuple_data=True,
    include_args=[0, 1, 2],
    include_kwargs=['data', 'gamma', 'lambda_']
)
def td_lambda_error(data: namedtuple, gamma: float = 0.9, lambda_: float = 0.8) -> torch.Tensor:
    """
    Overview:
        Computing TD(lambda) loss given constant gamma and lambda.
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (*including the terminal state*, values[terminal] should also be 0)
    Arguments:
        - data (:obj:`namedtuple`): td_lambda input data with fields ['value', 'reward', 'weight']
        - gamma (:obj:`float`): constant discount factor gamma, should be in [0, 1], defaults to 0.9
        - lambda (:obj:`float`): constant lambda, should be in [0, 1], defaults to 0.8
    Returns:
        - loss (:obj:`torch.Tensor`): Computed MSE loss, averaged over the batch
    Shapes:
        - value (:obj:`torch.FloatTensor`): :math:`(T+1, B)`, where T is trajectory length and B is batch,\
            which is the estimation of the state value at step 0 to T
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, the returns from time step 0 to T-1
        - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
        - loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
    """
    value, reward, weight = data
    if weight is None:
        weight = torch.ones_like(reward)
    with torch.no_grad():
        return_ = generalized_lambda_returns(value, reward, gamma, lambda_)
    # discard the value at T as it should be considered in the next slice
    loss = 0.5 * (F.mse_loss(return_, value[:-1], reduction='none') * weight).mean()
    return loss


def generalized_lambda_returns(
        bootstrap_values: torch.Tensor,
        rewards: torch.Tensor,
        gammas: float,
        lambda_: float,
        done: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Overview:
        Functional equivalent to trfl.value_ops.generalized_lambda_returns
        https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
        Passing in a number instead of tensor to make the value constant for all samples in batch
    Arguments:
        - bootstrap_values (:obj:`torch.Tensor` or :obj:`float`):
          estimation of the value at step 0 to *T*, of size [T_traj+1, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor` or :obj:`float`):
          discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - lambda (:obj:`torch.Tensor` or :obj:`float`): determining the mix of bootstrapping
          vs further accumulation of multistep returns at each timestep, of size [T_traj, batchsize]
        - done (:obj:`torch.Tensor` or :obj:`float`):
          whether the episode done at current step (from 0 to T-1), of size [T_traj, batchsize]
    Returns:
        - return (:obj:`torch.Tensor`): Computed lambda return value
          for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones_like(rewards)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    bootstrap_values_tp1 = bootstrap_values[1:, :]
    return multistep_forward_view(bootstrap_values_tp1, rewards, gammas, lambda_, done)


def multistep_forward_view(
        bootstrap_values: torch.Tensor,
        rewards: torch.Tensor,
        gammas: float,
        lambda_: float,
        done: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Overview:
        Same as trfl.sequence_ops.multistep_forward_view
        Implementing (12.18) in Sutton & Barto

        ```
        result[T-1] = rewards[T-1] + gammas[T-1] * bootstrap_values[T]
        for t in 0...T-2 :
        result[t] = rewards[t] + gammas[t]*(lambdas[t]*result[t+1] + (1-lambdas[t])*bootstrap_values[t+1])
        ```

        Assuming the first dim of input tensors correspond to the index in batch
    Arguments:
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the value at *step 1 to T*, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor`): discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - lambda (:obj:`torch.Tensor`): determining the mix of bootstrapping vs further accumulation of \
            multistep returns at each timestep of size [T_traj, batchsize], the element for T-1 is ignored \
            and effectively set to 0, as there is no information about future rewards.
        - done (:obj:`torch.Tensor` or :obj:`float`):
          whether the episode done at current step (from 0 to T-1), of size [T_traj, batchsize]
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value \
            for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    result = torch.empty_like(rewards)
    if done is None:
        done = torch.zeros_like(rewards)
    # Forced cutoff at the last one
    result[-1, :] = rewards[-1, :] + (1 - done[-1, :]) * gammas[-1, :] * bootstrap_values[-1, :]
    discounts = gammas * lambda_
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = rewards[t, :] + (1 - done[t, :]) * \
             (
                discounts[t, :] * result[t + 1, :] +
                (gammas[t, :] - discounts[t, :]) * bootstrap_values[t, :]
             )

    return result

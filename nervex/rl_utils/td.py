from collections import namedtuple
from typing import Union, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

q_1step_td_data = namedtuple('td_data', ['q', 'next_q', 'act', 'reward', 'done'])


def q_1step_td_error(
        data: namedtuple,
        gamma: float,
        weights: Optional[torch.Tensor] = None,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')  # noqa
) -> torch.Tensor:
    q, next_q, act, reward, done = data
    assert len(act.shape) == 1, act.shape
    assert len(reward.shape) == 1, reward.shape
    batch_range = torch.arange(act.shape[0])
    if weights is None:
        weights = torch.ones_like(reward)
    q_s_a = q[batch_range, act]
    next_act = next_q.argmax(dim=1)
    target_q_s_a = next_q[batch_range, next_act]
    target_q_s_a = gamma * (1 - done) * target_q_s_a + reward
    return (criterion(q_s_a, target_q_s_a.detach()) * weights).mean()


# q & next_q are List[torch.Tensor]
q_1step_td_data_continuous = namedtuple('td_data', ['q', 'next_q', 'act', 'reward', 'done'])


def q_1step_td_error_continuous(
        data: namedtuple,
        gamma: float,
        weights: Optional[torch.Tensor] = None,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')  # noqa
) -> List[torch.Tensor]:
    q, next_q, act, reward, done = data
    assert isinstance(q, list) and isinstance(next_q, list)
    assert len(act.shape) == 1, act.shape
    assert len(reward.shape) == 1, reward.shape
    batch_range = torch.arange(act.shape[0])
    if weights is None:
        weights = torch.ones_like(reward)
    q_s_a = [a_q[batch_range] for a_q in q]
    target_q_s_a = [a_next_q[batch_range] for a_next_q in next_q]
    target_q_s_a = min(target_q_s_a)
    target_q_s_a = gamma * (1 - done) * target_q_s_a + reward
    return [(criterion(a_q_s_a, target_q_s_a.detach()) * weights).mean() for a_q_s_a in q_s_a]


q_nstep_td_data = namedtuple('q_nstep_td_data', ['q', 'next_n_q', 'action', 'reward', 'done'])


def q_nstep_td_error(
        data: namedtuple,
        gamma: float,
        nstep: int = 1,
        weights: Optional[torch.Tensor] = None,
        criterion: torch.nn.modules = nn.MSELoss(reduction='none'),
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error for q-learning based algorithm
    Arguments:
        - data (:obj:`q_nstep_td_data`): the input data, q_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - weights (:obj:`torch.Tensor` or None): weights of td losses
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - nstep (:obj:`int`): nstep num, default set to 1
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
            ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
    """
    q, next_n_q, action, reward, done = data
    assert len(action.shape) == 1, action.shape
    if weights is None:
        weights = torch.ones_like(action)

    batch_range = torch.arange(action.shape[0])
    q_s_a = q[batch_range, action]
    next_n_act = next_n_q.argmax(dim=1)
    target_q_s_a = next_n_q[batch_range, next_n_act]

    reward_factor = torch.ones(nstep)
    for i in range(1, nstep):
        reward_factor[i] = gamma * reward_factor[i - 1]
    reward = torch.matmul(reward_factor, reward)

    target_q_s_a = reward + (gamma ** nstep) * target_q_s_a * (1 - done)
    return (criterion(q_s_a, target_q_s_a.detach()) * weights).mean()


td_lambda_data = namedtuple('td_lambda_data', ['value', 'reward', 'weight'])


def td_lambda_error(data: namedtuple, gamma: float = 0.9, lambda_: float = 0.8) -> torch.Tensor:
    """
    Overview:
        Computing TD($\lambda$) loss given constant gamma and lambda.
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (*including the terminal state*, values[terminal] should also be 0)
    Arguments:
        - data (:obj:`namedtuple`): td_lambda input data with fields ['value', 'reward', 'weight']
        - gamma (:obj:`float`): constant discount factor gamma, should be in [0, 1], defaults to 0.9
        - lambda_ (:obj:`float`): constant lambda, should be in [0, 1], defaults to 0.8
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
        bootstrap_values: torch.Tensor, rewards: torch.Tensor, gammas: float, lambda_: float
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
        - lambda_ (:obj:`torch.Tensor` or :obj:`float`): determining the mix of bootstrapping
          vs further accumulation of multistep returns at each timestep, of size [T_traj, batchsize]
    Returns:
        - return_ (:obj:`torch.Tensor`): Computed lambda return value
          for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones_like(rewards)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    bootstrap_values_tp1 = bootstrap_values[1:, :]
    return multistep_forward_view(bootstrap_values_tp1, rewards, gammas, lambda_)


def multistep_forward_view(
        bootstrap_values: torch.Tensor, rewards: torch.Tensor, gammas: float, lambda_: float
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
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (including the terminal state, which is, bootstrap_values[terminal] should also be 0)
    Arguments:
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the value at *step 1 to T*, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor`): discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - lambda_ (:obj:`torch.Tensor`): determining the mix of bootstrapping
        vs further accumulation of multistep returns at each timestep of size [T_traj, batchsize],
        the element for T-1 is ignored and effectively set to 0,
        as there is no information about future rewards.
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value
         for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    result = torch.empty_like(rewards)
    # Forced cutoff at the last one
    result[-1, :] = rewards[-1, :] + gammas[-1, :] * bootstrap_values[-1, :]
    discounts = gammas * lambda_
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = rewards[t, :] \
                       + discounts[t, :] * result[t + 1, :] \
                       + (gammas[t, :] - discounts[t, :]) * bootstrap_values[t, :]

    return result

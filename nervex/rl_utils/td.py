from collections import namedtuple
from typing import Union

import torch
import torch.nn as nn

td_data = namedtuple('td_data', ['q', 'next_q', 'act', 'reward', 'terminate'])


def one_step_td_error(
        data: td_data,
        gamma: float,
        weights: Union[torch.Tensor, None],
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')  # noqa
) -> torch.Tensor:
    q, next_q, act, reward, terminate = data
    assert len(reward.shape) == 1
    batch_range = torch.arange(act.shape[0])
    if weights is None:
        weights = torch.ones_like(reward)

    q_s_a = q[batch_range, act]

    next_act = next_q.argmax(dim=1)
    target_q_s_a = next_q[batch_range, next_act]
    target_q_s_a = gamma * (1 - terminate) * target_q_s_a + reward
    return (criterion(q_s_a, target_q_s_a.detach()) * weights).mean()


td_data_a2c = namedtuple('td_data_a2c', ['v', 'next_v', 'reward', 'terminate'])


def one_step_td_error_a2c(
        data: td_data_a2c,
        gamma: float,
        weights: Union[torch.Tensor, None],
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')  # noqa
) -> torch.Tensor:
    v, next_v, reward, terminate = data
    v, next_v = v.squeeze(), next_v.squeeze()
    if weights is None:
        weights = torch.ones_like(reward, dtype=torch.float32)
    reward = reward
    target_v = gamma * (1 - terminate) * next_v + reward
    mean_loss = (criterion(v, target_v.detach()) * weights).mean()
    return mean_loss

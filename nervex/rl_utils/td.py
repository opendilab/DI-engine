from collections import namedtuple
from typing import Union
import torch
import torch.nn as nn

td_data = namedtuple('td_data', ['q', 'next_q', 'act', 'reward', 'terminate'])


def one_step_td_error(
        data: td_data,
        gamma: float,
        weights: Union[torch.Tensor, None],
        criterion: torch.nn.modules = nn.MSELoss(reduction='none')
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

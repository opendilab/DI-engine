from collections import namedtuple
from typing import Union, Optional, List

import torch
import torch.nn.functional as F

# compute soft q loss and value loss for Soft Actor Critic

soft_q_data = namedtuple('soft_q_data', ['target_v_value', 'reward', 'done', 'q_value'])


def soft_q_error(data: namedtuple, discount) -> torch.Tensor:
    target_v_value, reward, done, q_value = data
    target_q_value = reward + (1. - done) * discount * target_v_value
    q_loss = F.mse_loss(q_value, target_q_value.detach())
    return q_loss


value_data = namedtuple('soft_q_data', ['v_value', 'next_v_value'])


def value_error(data: namedtuple) -> torch.Tensor:
    v_value, next_v_value = data
    value_loss = F.mse_loss(v_value, next_v_value.detach())
    return value_loss

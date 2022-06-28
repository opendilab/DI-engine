"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/model.py
"""

import torch
import numpy as np
import torch.nn as nn
from typing import List, NamedTuple
from dataclasses import dataclass
from ding.rl_utils.mcts.utils import mask_nan


@dataclass
class NetworkOutput:
    # output format of the model
    value: float
    value_prefix: float
    policy_logits: List[float]
    hidden_state: List[float]
    reward_hidden: object


class BaseNet(nn.Module):

    def __init__(self, lstm_hidden_size):
        """
        Overview:
            Base Network
        Arguments：
            - lstm_hidden_size: int dim of lstm hidden
        """
        super(BaseNet, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        num = obs.size(0)
        hidden_state = self.representation(obs)
        policy_logits, value = self.prediction(hidden_state)
        # zero initialization for reward hidden states
        reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size), torch.zeros(1, num, self.lstm_hidden_size))
        return NetworkOutput(value, [0. for _ in range(num)], policy_logits, hidden_state, reward_hidden)

    def recurrent_inference(
            self, hidden_state: torch.Tensor, reward_hidden: torch.Tensor, action: torch.Tensor
    ) -> NetworkOutput:
        hidden_state, reward_hidden, value_prefix = self.dynamics(hidden_state, reward_hidden, action)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, value_prefix, policy_logits, hidden_state, reward_hidden)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients: torch.Tensor):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = g


class DiscreteSupport(object):

    def __init__(self, min: int, max: int, delta=1.):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + 1, delta)
        self.size = len(self.range)
        self.delta = delta


def inverse_scalar_transform(logits, support_size):
    """ Reference from MuZero: Appendix F => Network Architecture
    & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    """
    scalar_support = DiscreteSupport(-support_size, support_size, delta=1)
    delta = scalar_support.delta
    value_probs = torch.softmax(logits, dim=1)
    value_support = torch.ones(value_probs.shape)
    value_support[:, :] = torch.from_numpy(np.array([x for x in scalar_support.range]))
    value_support = value_support.to(device=value_probs.device)
    value = (value_support * value_probs).sum(1, keepdim=True) / delta

    epsilon = 0.001
    sign = torch.ones(value.shape).float().to(value.device)
    sign[value < 0] = -1.0
    output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
    output = sign * output * delta
    output = mask_nan(output)
    output[torch.abs(output) < epsilon] = 0.

    return output

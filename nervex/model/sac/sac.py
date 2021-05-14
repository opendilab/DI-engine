import sys
import os

import math
import random
from functools import partial
from typing import Union, List, Dict, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from nervex.utils import squeeze, MODEL_REGISTRY
from nervex.torch_utils import MLP
from ..common import ActorCriticBase


class SoftQNet(nn.Module):

    def __init__(self, obs_shape, action_shape, soft_q_hidden_size: int, init_w: float = 3e-3):
        super(SoftQNet, self).__init__()
        self._act = nn.ReLU()

        input_dim = squeeze(obs_shape + action_shape)
        output_layer = nn.Linear(soft_q_hidden_size, 1)
        output_layer.weight.data.uniform_(-init_w, init_w)
        output_layer.bias.data.uniform_(-init_w, init_w)
        self._main = nn.Sequential(MLP(input_dim, 256, soft_q_hidden_size, 2, activation=self._act), output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        return x


class ValueNet(nn.Module):

    def __init__(self, obs_shape, value_hidden_size: int, init_w: float = 3e-3):
        super(ValueNet, self).__init__()
        self._act = nn.ReLU()
        input_dim = squeeze(obs_shape)
        output_layer = nn.Linear(value_hidden_size, 1)
        output_layer.weight.data.uniform_(-init_w, init_w)
        output_layer.bias.data.uniform_(-init_w, init_w)
        self._main = nn.Sequential(MLP(input_dim, 256, value_hidden_size, 2, activation=self._act), output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        return x


class PolicyNet(nn.Module):

    def __init__(
        self,
        obs_shape,
        action_shape,
        policy_hidden_size: int,
        log_std_min: int = -20,
        log_std_max: int = 2,
        init_w: float = 3e-3
    ):
        super(PolicyNet, self).__init__()
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self._act = nn.ReLU()
        input_dim = squeeze(obs_shape)
        output_dim = squeeze(action_shape)

        self._main = MLP(input_dim, 256, policy_hidden_size, 2, activation=self._act)

        self._mean_layer = nn.Linear(policy_hidden_size, output_dim)
        self._mean_layer.weight.data.uniform_(-init_w, init_w)
        self._mean_layer.bias.data.uniform_(-init_w, init_w)

        self._log_std_layer = nn.Linear(policy_hidden_size, output_dim)
        self._log_std_layer.weight.data.uniform_(-init_w, init_w)
        self._log_std_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        mean = self._mean_layer(x)
        log_std = self._log_std_layer(x)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        return mean, log_std


@MODEL_REGISTRY.register('sac')
class SAC(ActorCriticBase):

    def __init__(
            self,
            obs_shape: tuple,
            action_shape: Union[int, tuple],
            policy_embedding_size: int = 128,
            value_embedding_size: int = 128,
            soft_q_embedding_size: int = 128,
            twin_q: bool = True,
            value_network: bool = False,
    ) -> None:
        super(SAC, self).__init__()
        self.modes.append('evaluate')

        self._act = nn.ReLU()
        self._value_network = value_network
        # input info
        self._obs_shape: int = squeeze(obs_shape)
        self._act_shape: int = squeeze(action_shape)

        # embedding info
        self._policy_embedding_size: int = policy_embedding_size
        self._value_embedding_size: int = value_embedding_size
        self._soft_q_embedding_size: int = soft_q_embedding_size

        # build network
        if self._value_network:
            self._value_net = ValueNet(self._obs_shape, self._value_embedding_size)
        self._policy_net = PolicyNet(self._obs_shape, self._act_shape, self._policy_embedding_size)

        self._twin_q = twin_q
        if not self._twin_q:
            self._soft_q_net = SoftQNet(self._obs_shape, self._act_shape, self._soft_q_embedding_size)
        else:
            self._soft_q_net = nn.ModuleList()
            for i in range(2):
                self._soft_q_net.append(SoftQNet(self._obs_shape, self._act_shape, self._soft_q_embedding_size))

    def _value_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._value_network
        return self._value_net(x).squeeze(1)

    def _soft_q_net_forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self._twin_q:
            return self._soft_q_net(x).squeeze(1)
        else:
            q_value = []
            for i in range(2):
                q_value.append(self._soft_q_net[i](x).squeeze(1))
            return q_value

    def _policy_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy_net(x)

    def compute_critic_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        action = inputs['action']
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        state_action_input = torch.cat([inputs['obs'], action], dim=1)
        q_value = self._soft_q_net_forward(state_action_input)
        return {'q_value': q_value}

    def compute_critic_v(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_value = self._value_net_forward(obs)
        return {'v_value': v_value}

    def compute_critic(self, inputs: Dict[str, torch.Tensor], qv='q') -> Dict[str, torch.Tensor]:
        assert qv in ['q', 'v'], qv
        if 'q' == qv:
            return self.compute_critic_q(inputs)
        else:
            return self.compute_critic_v(inputs)

    def compute_actor(self, obs: torch.Tensor, deterministic_eval=False, epsilon=1e-6) -> Dict[str, torch.Tensor]:
        mean, log_std = self._policy_net_forward(obs)
        std = log_std.exp()

        dist = Independent(Normal(mean, std), 1)
        # for reparameterization trick (mean + std * N(0,1))
        if deterministic_eval:
            x = mean
        else:
            x = dist.rsample()
        y = torch.tanh(x)
        action = y

        # epsilon is used to avoid log of zero/negative number.
        y = 1 - y.pow(2) + epsilon
        log_prob = dist.log_prob(x).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        return {'mean': mean, 'log_std': log_std, 'action': action, 'log_prob': log_prob}

    @property
    def policy_net(self) -> torch.nn.Module:
        return self._policy_net

    @property
    def value_net(self) -> torch.nn.Module:
        assert self._value_network
        return self._value_net

    @property
    def q_net(self) -> torch.nn.Module:
        return self._soft_q_net

import sys
import os

import math
import random
from functools import partial
from typing import Union, List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.distributions import Normal

from nervex.model.common_arch import SoftActorCriticBase
from nervex.utils import squeeze


class SoftQNet(nn.Module):

    def __init__(self, obs_dim, action_dim, soft_q_hidden_dim: int):
        super(SoftQNet, self).__init__()
        self._act = nn.ReLU()
        input_dim = squeeze(obs_dim + action_dim)
        hidden_dim = soft_q_hidden_dim
        hidden_dim_list = [128, 128] + [hidden_dim] + [1]

        layers = []
        for dim in hidden_dim_list:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self._act)
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self._main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        return x


class ValueNet(nn.Module):

    def __init__(self, obs_dim, value_hidden_dim: int):
        super(ValueNet, self).__init__()
        self._act = nn.ReLU()
        input_dim = squeeze(obs_dim)
        hidden_dim = value_hidden_dim
        hidden_dim_list = [128, 128] + [hidden_dim] + [1]

        layers = []
        for dim in hidden_dim_list:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self._act)
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self._main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        return x


class PolicyNet(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim: int,
        policy_hidden_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 2,
        init_w: float = 3e-3
    ):
        super(PolicyNet, self).__init__()
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self._act = nn.ReLU()
        hidden_dim = policy_hidden_dim
        input_dim = squeeze(obs_dim)
        hidden_dim_list = [128, 128] + [hidden_dim]

        layers = []
        for dim in hidden_dim_list:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self._act)
            input_dim = dim
        self._main = nn.Sequential(*layers)

        self._mean_layer = nn.Linear(hidden_dim, action_dim)
        self._mean_layer.weight.data.uniform_(-init_w, init_w)
        self._mean_layer.bias.data.uniform_(-init_w, init_w)

        self._log_std_layer = nn.Linear(hidden_dim, action_dim)
        self._log_std_layer.weight.data.uniform_(-init_w, init_w)
        self._log_std_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        mean = self._mean_layer(x)
        log_std = self._log_std_layer(x)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        return mean, log_std


class SAC(SoftActorCriticBase):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, tuple],
            policy_embedding_dim: int = 64,
            value_embedding_dim: int = 64,
            soft_q_embedding_dim: int = 64,
            use_twin_q: bool = False
    ) -> None:
        super(SAC, self).__init__()

        self._act = nn.ReLU()
        # input info
        self._obs_dim: int = squeeze(obs_dim)
        self._act_dim: int = squeeze(action_dim)

        # embedding info
        self._policy_embedding_dim: int = policy_embedding_dim
        self._value_embedding_dim: int = value_embedding_dim
        self._soft_q_embedding_dim: int = soft_q_embedding_dim

        # build network
        self._policy_net = PolicyNet(self._obs_dim, self._act_dim, self._policy_embedding_dim)
        self._value_net = ValueNet(self._obs_dim, self._value_embedding_dim)

        self._use_twin_q = use_twin_q
        if not self._use_twin_q:
            self._soft_q_net = SoftQNet(self._obs_dim, self._act_dim, self._soft_q_embedding_dim)
        else:
            self._soft_q_net = nn.ModuleList()
            for i in range(2):
                self._soft_q_net[i] = SoftQNet(self._obs_dim, self._act_dim, self._soft_q_embedding_dim)

    def _value_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._value_net(x).squeeze(1)

    def _soft_q_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_twin_q:
            return self._soft_q_net(x).squeeze(1)
        else:
            q_value = []
            for i in range(2):
                q_value.append(self._soft_q_net[i](x))
            return q_value

    def _policy_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy_net(x)

    def compute_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        action = inputs['action']
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        state_action_input = torch.cat([inputs['obs'], action], dim=1)
        q_value = self._soft_q_net_forward(state_action_input)
        if self._use_twin_q:
            return {'q_value': min(q_value)}
        else:
            return {'q_value': q_value}

    def compute_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        v_value = self._value_net_forward(state_input)
        return {'v_value': v_value}

    def compute_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        mean, log_std = self._policy_net_forward(state_input)
        std = log_std.exp()

        dist = Normal(mean, std)
        z = dist.sample()
        action = torch.tanh(z)

        if action.shape[1] == 1:
            action = action.squeeze(1)

        return {'action': action}


# use tanh action

    def evaluate(self, inputs: Dict[str, torch.Tensor], epsilon=1e-6) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        mean, log_std = self._policy_net_forward(state_input)
        std = log_std.exp()

        dist = Normal(mean, std)
        z = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(z)
        if action.shape[1] == 1:
            action = action.squeeze(1)

        log_prob = dist.log_prob(z)
        # enforcing action bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return {'mean': mean, 'log_std': log_std, 'action': action, 'log_prob': log_prob}

    @property
    def policy_net(self) -> torch.nn.Module:
        return self._policy_net

    @property
    def value_net(self) -> torch.nn.Module:
        return self._value_net

    @property
    def q_net(self) -> torch.nn.Module:
        return self._soft_q_net

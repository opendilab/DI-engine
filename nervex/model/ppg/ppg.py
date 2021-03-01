import sys
import os

import math
import random
from functools import partial
from typing import Union, List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.distributions import Normal

from nervex.utils import squeeze
from ..common import PhasicPolicyGradientBase, FCEncoder, register_model
from ..actor_critic.value_ac import FCValueAC


class PolicyNet(nn.Module):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            embedding_dim: int,
            head_hidden_dim: int = 128,
            continous=False,
            fixed_sigma_value=None,
    ) -> None:
        r"""
        Overview:
            Init the ValueAC according to arguments.
        Arguments:
            - obs_dim (:obj:`tuple`): a tuple of observation dim
            - action_dim (:obj:`int`): the num of actions
            - embedding_dim (:obj:`int`): encoder's embedding dim (output dim)
            - head_hidden_dim (:obj:`int`): the hidden dim in actor and critic heads
        """
        super(PolicyNet, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._act_dim = squeeze(action_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2
        self.continous = continous
        self.fixed_sigma_value = fixed_sigma_value
        # actor head
        input_dim = embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_dim, head_hidden_dim))
            layers.append(self._act)
            input_dim = head_hidden_dim
        layers.append(nn.Linear(input_dim, self._act_dim))
        self._actor = nn.Sequential(*layers)
        # sigma head
        if continous and self.fixed_sigma_value is None:
            input_dim = embedding_dim
            layers = []
            for _ in range(self._head_layer_num):
                layers.append(nn.Linear(input_dim, head_hidden_dim))
                layers.append(self._act)
                input_dim = head_hidden_dim
            layers.append(nn.Linear(input_dim, self._act_dim))
            self._log_sigma = nn.Sequential(*layers)
        # critic head
        input_dim = embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_dim, head_hidden_dim))
            layers.append(self._act)
            input_dim = head_hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self._critic = nn.Sequential(*layers)

    def _setup_encoder(self) -> torch.nn.Module:
        r"""
        Overview:
            Setup an ``ConvEncoder`` to encode 2-dim observation
        Returns:
            - encoder (:obj:`torch.nn.Module`): ``ConvEncoder``
        """
        return FCEncoder(self._obs_dim, self._embedding_dim)

    def _critic_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Use critic head to output value of current state.
        Arguments:
            - x (:obj:`torch.Tensor`): embedding tensor after encoder
        """
        return self._critic(x).squeeze(1)

    def _actor_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Use actor head to output q-value of n-dim discrete actions.
        Arguments:
            - x (:obj:`torch.Tensor`): embedding tensor after encoder
        """
        return self._actor(x)

    def compute_action_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            First encode raw observation, then output value and logit.
            Normal reinforcement learning training, often called by learner to optimize both critic and actor.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): embedding tensor after encoder
        Returns:
            - ret (:obj:`Dict[str, torch.Tensor]`): a dict containing value and logit
        """
        # for compatible, but we recommend use dict as input format
        if isinstance(inputs, torch.Tensor):
            embedding = self._encoder(inputs)
        else:
            embedding = self._encoder(inputs['obs'])
        value = self._critic_forward(embedding)
        logit = self._actor_forward(embedding)
        if self.continous:
            mu = torch.tanh(logit)
            sigma = torch.clamp_max(
                self._log_sigma(embedding), max=2
            ).exp(
            ) if self.fixed_sigma_value is None else self.fixed_sigma_value * torch.ones_like(mu)  # fix gamma to debug
            logit = (mu, sigma)

        return {'value': value, 'logit': logit}

    def compute_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            First encode raw observation, then output logit.
            Evaluate policy performance only using the actor part, often called by evaluator.
        Arguments:
            - x (:obj:`torch.Tensor`): embedding tensor after encoder
        Returns:
            - ret (:obj:`Dict[str, torch.Tensor]`): a dict containing only logit
        """
        if isinstance(inputs, torch.Tensor):
            embedding = self._encoder(inputs)
        else:
            embedding = self._encoder(inputs['obs'])
        logit = self._actor_forward(embedding)
        if self.continous:
            mu = torch.tanh(logit)
            sigma = torch.clamp_max(
                self._log_sigma(embedding), max=2
            ).exp(
            ) if self.fixed_sigma_value is None else self.fixed_sigma_value * torch.ones_like(mu)  # fix gamma to debug
            logit = (mu, sigma)

        return {'logit': logit}

    def compute_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            First encode raw observation, then output value and logit.
            Normal reinforcement learning training, often called by learner to optimize both critic and actor.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): embedding tensor after encoder
        Returns:
            - ret (:obj:`Dict[str, torch.Tensor]`): a dict containing value and logit
        """
        # for compatible, but we recommend use dict as input format
        if isinstance(inputs, torch.Tensor):
            embedding = self._encoder(inputs)
        else:
            embedding = self._encoder(inputs['obs'])
        value = self._critic_forward(embedding)

        return {'value': value}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            First encode raw observation, then output value and logit.
            Normal reinforcement learning training, often called by learner.rst to optimize both critic and actor.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): embedding tensor after encoder
        Returns:
            - ret (:obj:`Dict[str, torch.Tensor]`): a dict containing value and logit
        """
        # for compatible, but we recommend use dict as input format
        # if isinstance(inputs, torch.Tensor):
        #     embedding = self._encoder(inputs)
        # else:
        #     embedding = self._encoder(inputs['obs'])
        rets = self.compute_action_value(inputs)
        value = rets['value']
        logit = rets['logit']
        return value, logit


class ValueNet(nn.Module):

    def __init__(
            self,
            obs_dim: tuple,
            embedding_dim: int,
            head_hidden_dim: int = 128,
    ) -> None:
        super(ValueNet, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2

        # critic head
        input_dim = embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_dim, head_hidden_dim))
            layers.append(self._act)
            input_dim = head_hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self._critic = nn.Sequential(*layers)

    def _setup_encoder(self) -> torch.nn.Module:
        r"""
        Overview:
            Setup an ``ConvEncoder`` to encode 2-dim observation
        Returns:
            - encoder (:obj:`torch.nn.Module`): ``ConvEncoder``
        """
        return FCEncoder(self._obs_dim, self._embedding_dim)

    def _critic_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Use critic head to output value of current state.
        Arguments:
            - x (:obj:`torch.Tensor`): embedding tensor after encoder
        """
        return self._critic(x).squeeze(1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            First encode raw observation, then output value and logit.
            Normal reinforcement learning training, often called by learner.rst to optimize both critic and actor.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): embedding tensor after encoder
        Returns:
            - ret (:obj:`Dict[str, torch.Tensor]`): a dict containing value and logit
        """
        # for compatible, but we recommend use dict as input format
        if isinstance(inputs, torch.Tensor):
            embedding = self._encoder(inputs)
        else:
            embedding = self._encoder(inputs['obs'])
        value = self._critic_forward(embedding)

        return value


class PPG(PhasicPolicyGradientBase):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, tuple],
            # policy_embedding_dim: int = 128,
            # value_embedding_dim: int = 128
        embedding_dim: int = 128
    ) -> None:
        super(PPG, self).__init__()

        self._act = nn.ReLU()
        # input info
        self._obs_dim: int = squeeze(obs_dim)
        self._act_dim: int = squeeze(action_dim)

        # embedding info
        policy_embedding_dim = embedding_dim
        value_embedding_dim = embedding_dim
        self._policy_embedding_dim: int = policy_embedding_dim
        self._value_embedding_dim: int = value_embedding_dim

        # build network
        self._policy_net = PolicyNet(self._obs_dim, self._act_dim, self._policy_embedding_dim)
        self._value_net = ValueNet(self._obs_dim, self._value_embedding_dim)

    def _value_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._value_net(x)

    def _policy_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy_net(x)

    def compute_action_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        value = self._value_net_forward(state_input)
        logit = self._policy_net.compute_action(state_input)['logit']
        return {'logit': logit, 'value': value}

    def compute_action(self,
                       inputs: Dict[str, torch.Tensor],
                       deterministic_eval: bool = False) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        logit = self._policy_net.compute_action(state_input)['logit']
        return {'logit': logit}

    def compute_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        value = self._value_net_forward(state_input)
        return {'value': value}

    def compute_policy_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        value = self._policy_net.compute_value(state_input)['value']
        return {'value': value}

    @property
    def policy_net(self) -> torch.nn.Module:
        return self._policy_net

    @property
    def value_net(self) -> torch.nn.Module:
        return self._value_net


register_model('ppg', PPG)

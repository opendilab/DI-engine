import sys
import os

import math
import random
from functools import partial
from typing import Union, List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.distributions import Normal

from nervex.utils import squeeze, MODEL_REGISTRY
from nervex.torch_utils.network.nn_module import MLP
from ..common import ActorCriticBase, FCEncoder, ConvEncoder
from ..actor_critic.value_ac import FCValueAC, ConvValueAC

class FCValueNet(nn.Module):

    def __init__(
            self,
            obs_dim: tuple,
            embedding_dim: int,
            head_hidden_dim: int = 128,
    ) -> None:
        super(FCValueNet, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2

        # critic head
        self._critic = nn.Sequential(MLP(embedding_dim, head_hidden_dim, head_hidden_dim, self._head_layer_num, activation=self._act),
                                        nn.Linear(head_hidden_dim, 1))

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


class ConvValueNet(nn.Module):

    def __init__(
            self,
            obs_dim: tuple,
            embedding_dim: int,
            head_hidden_dim: int = 128,
    ) -> None:
        super(ConvValueNet, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2

        # critic head
        self._critic = nn.Sequential(MLP(embedding_dim, head_hidden_dim, head_hidden_dim, self._head_layer_num, activation=self._act),
                                        nn.Linear(head_hidden_dim, 1))

    def _setup_encoder(self) -> torch.nn.Module:
        r"""
        Overview:
            Setup an ``ConvEncoder`` to encode 2-dim observation
        Returns:
            - encoder (:obj:`torch.nn.Module`): ``ConvEncoder``
        """
        return ConvEncoder(self._obs_dim, self._embedding_dim)

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


@MODEL_REGISTRY.register('fc_ppg')
class FCPPG(ActorCriticBase):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, tuple],
            embedding_dim: int = 128,
            # policy_embedding_dim: int = 128,
            # value_embedding_dim: int = 128
    ) -> None:
        super(FCPPG, self).__init__()

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
        self._policy_net = FCValueAC(self._obs_dim, self._act_dim, self._policy_embedding_dim)
        self._value_net = FCValueNet(self._obs_dim, self._value_embedding_dim)

    def _value_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._value_net(x)

    def _policy_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy_net(x)

    def compute_actor_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        return self._policy_net.compute_actor_critic(state_input)

    def compute_actor(self,
                       inputs: Dict[str, torch.Tensor],
                       deterministic_eval: bool = False) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        logit = self._policy_net.compute_actor(state_input)['logit']
        return {'logit': logit}

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        value = self._value_net_forward(state_input)
        return {'value': value}

    @property
    def policy_net(self) -> torch.nn.Module:
        return self._policy_net

    @property
    def value_net(self) -> torch.nn.Module:
        return self._value_net


@MODEL_REGISTRY.register('conv_ppg')
class ConvPPG(ActorCriticBase):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, tuple],
            embedding_dim: int = 128,
            # policy_embedding_dim: int = 128,
            # value_embedding_dim: int = 128
    ) -> None:
        super(ConvPPG, self).__init__()

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
        self._policy_net = ConvValueAC(self._obs_dim, self._act_dim, self._policy_embedding_dim)
        self._value_net = ConvValueNet(self._obs_dim, self._value_embedding_dim)

    def _value_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._value_net(x)

    def _policy_net_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy_net(x)

    def compute_actor_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        return self._policy_net.compute_action_value(state_input)

    def compute_actor(self,
                       inputs: Dict[str, torch.Tensor],
                       deterministic_eval: bool = False) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        logit = self._policy_net.compute_action(state_input)['logit']
        return {'logit': logit}

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        value = self._value_net_forward(state_input)
        return {'value': value}

    @property
    def policy_net(self) -> torch.nn.Module:
        return self._policy_net

    @property
    def value_net(self) -> torch.nn.Module:
        return self._value_net

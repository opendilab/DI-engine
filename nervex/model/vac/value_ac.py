from typing import Dict, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nervex.utils import squeeze, MODEL_REGISTRY
from ..common import ActorCriticBase, ConvEncoder, FCEncoder


class ValueAC(ActorCriticBase):
    r"""
    Overview:
        Actor-Critic model. Critic part outputs value of current state,
        and that is why it is called "ValueAC", which is in comparison with "QAC".
        Actor part outputs the size of N probability of selecting corresponding discrete action.
        It is the model adopted in A2C.
    Interface:
        __init__, forward, set_seed, compute_action_value, compute_action
    """

    def __init__(
            self,
            obs_shape: tuple,
            action_shape: Union[int, list],
            embedding_size: int,
            head_hidden_size: int = 128,
            continous=False,
            fixed_sigma_value=None,
    ) -> None:
        r"""
        Overview:
            Init the ValueAC according to arguments.
        Arguments:
            - obs_shape (:obj:`tuple`): a tuple of observation sh
            - action_shape (:obj:`Union[int, list]`): the num of actions
            - embedding_size (:obj:`int`): encoder's embedding size (output size)
            - head_hidden_size (:obj:`int`): the hidden size in actor and critic heads
        """
        super(ValueAC, self).__init__()
        self._act = nn.ReLU()
        self._obs_shape = squeeze(obs_shape)
        self._act_shape = squeeze(action_shape)
        self._embedding_size = embedding_size
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2
        self._head_hidden_size = head_hidden_size
        self.continous = continous
        self.fixed_sigma_value = fixed_sigma_value
        # actor head
        if isinstance(self._act_shape, tuple):
            self._actor = nn.ModuleList([self._setup_actor(a) for a in self._act_shape])
        else:
            self._actor = self._setup_actor(self._act_shape)
        # sigma head
        if continous and self.fixed_sigma_value is None:
            input_size = embedding_size
            layers = []
            for _ in range(self._head_layer_num):
                layers.append(nn.Linear(input_size, head_hidden_size))
                layers.append(self._act)
                input_size = head_hidden_size
            layers.append(nn.Linear(input_size, self._act_shape))
            self._log_sigma = nn.Sequential(*layers)
        # critic head
        input_size = embedding_size
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_size, head_hidden_size))
            layers.append(self._act)
            input_size = head_hidden_size
        layers.append(nn.Linear(input_size, 1))

        self._critic = nn.Sequential(*layers)

    def _setup_actor(self, act_shape: int) -> torch.nn.Module:
        input_size = self._embedding_size
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_size, self._head_hidden_size))
            layers.append(self._act)
            input_size = self._head_hidden_size
        layers.append(nn.Linear(input_size, act_shape))
        return nn.Sequential(*layers)

    def _setup_encoder(self) -> torch.nn.Module:
        r"""
        Overview:
            Setup the encoder to encode env's raw observation
        """
        raise NotImplementedError

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
            Use actor head to output q-value of the size of N discrete actions.
        Arguments:
            - x (:obj:`torch.Tensor`): embedding tensor after encoder
        """
        if isinstance(self._act_shape, tuple):
            return [m(x) for m in self._actor]
        else:
            return self._actor(x)

    def compute_actor_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            First encode raw observation, then output value and logit.
            Normal reinforcement learning training, often called by learner to optimize both critic and actor.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): embedding tensor after encoder
        Returns:
            - ret (:obj:`Dict[str, torch.Tensor]`): a dict containing value and logit
        """
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

    def compute_actor(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


@MODEL_REGISTRY.register('conv_vac')
class ConvValueAC(ValueAC):
    r"""
    Overview:
        Convolution Actor-Critic model. Encode the observation with a ``ConvEncoder``
    Interface:
        __init__, forward, compute_action_value, compute_action
    """

    def _setup_encoder(self) -> torch.nn.Module:
        r"""
        Overview:
            Setup an ``ConvEncoder`` to encode 2-dim observation
        Returns:
            - encoder (:obj:`torch.nn.Module`): ``ConvEncoder``
        """
        return ConvEncoder(self._obs_shape, self._embedding_size)


@MODEL_REGISTRY.register('fc_vac')
class FCValueAC(ValueAC):
    r"""
    Overview:
        Convolution Actor-Critic model. Encode the observation with a ``FCEncoder``
    Interface:
        __init__, forward, compute_action_value, compute_action
    """

    def _setup_encoder(self) -> torch.nn.Module:
        r"""
        Overview:
            Setup an ``FCEncoder`` to encode 1-dim observation
        Returns:
            - encoder (:obj:`torch.nn.Module`): ``ConvEncoder``
        """
        return FCEncoder(self._obs_shape, self._embedding_size)

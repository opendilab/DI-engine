import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Optional
from nervex.utils import squeeze, MODEL_REGISTRY
from ..common import ValueActorCriticBase, ConvEncoder, FCEncoder


class ValueAC(ValueActorCriticBase):
    r"""
    Overview:
        Actor-Critic model. Critic part outputs value of current state,
        and that is why it is called "ValueAC", which is in comparison with "QAC".
        Actor part outputs n-dim probability of selecting corresponding discrete action.
        It is the model adopted in A2C.
    Interface:
        __init__, forward, set_seed, compute_action_value, compute_action
    """

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, list],
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
            - action_dim (:obj:`Union[int, list]`): the num of actions
            - embedding_dim (:obj:`int`): encoder's embedding dim (output dim)
            - head_hidden_dim (:obj:`int`): the hidden dim in actor and critic heads
        """
        super(ValueAC, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._act_dim = squeeze(action_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2
        self._head_hidden_dim = head_hidden_dim
        self.continous = continous
        self.fixed_sigma_value = fixed_sigma_value
        # actor head
        if isinstance(self._act_dim, tuple):
            self._actor = nn.ModuleList([self._setup_actor(a) for a in self._act_dim])
        else:
            self._actor = self._setup_actor(self._act_dim)
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

    def _setup_actor(self, act_dim: int) -> torch.nn.Module:
        input_dim = self._embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_dim, self._head_hidden_dim))
            layers.append(self._act)
            input_dim = self._head_hidden_dim
        layers.append(nn.Linear(input_dim, act_dim))
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
            Use actor head to output q-value of n-dim discrete actions.
        Arguments:
            - x (:obj:`torch.Tensor`): embedding tensor after encoder
        """
        if isinstance(self._act_dim, tuple):
            return [m(x) for m in self._actor]
        else:
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
        return ConvEncoder(self._obs_dim, self._embedding_dim)


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
            Setup an ``ConvEncoder`` to encode 2-dim observation
        Returns:
            - encoder (:obj:`torch.nn.Module`): ``ConvEncoder``
        """
        return FCEncoder(self._obs_dim, self._embedding_dim)

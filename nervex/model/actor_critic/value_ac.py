import torch
import torch.nn as nn
from typing import Dict
from ..common_arch import ValueActorCriticBase, ConvEncoder, FCEncoder
from nervex.utils import squeeze


class ValueAC(ValueActorCriticBase):
    r"""
    Overview:
        Actor-Critic model. Critic part outputs value of current state, and that is why it is called "ValueAC"
        Actor part outputs n-dim q value of discrete actions.
        It is the model which is adopted in A2C.
    Interface:
        __init__, forward, set_seed, compute_action_value, compute_action
    """

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            embedding_dim: int,
            head_hidden_dim: int = 128,
            continous=False,
            act_limit=1.0
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
        super(ValueAC, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._act_dim = squeeze(action_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2
        self.continous = continous
        self.act_limit = act_limit
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
        if continous:
            input_dim = embedding_dim
            layers = []
            for _ in range(self._head_layer_num):
                layers.append(nn.Linear(input_dim, head_hidden_dim))
                layers.append(self._act)
                input_dim = head_hidden_dim
            layers.append(nn.Linear(input_dim, self._act_dim))
            self._sigma = nn.Sequential(*layers)
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
            mu = self.act_limit * torch.tanh(logit)
            sigma = self._sigma(embedding)
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
            logit = self.act_limit * torch.tanh(logit)
        return {'logit': logit}


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

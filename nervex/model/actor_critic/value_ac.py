import torch
import torch.nn as nn
from typing import Dict
from ..common_arch import ActorCriticBase, ConvEncoder
from nervex.utils import squeeze


class ValueAC(ActorCriticBase):

    def __init__(self, obs_dim: tuple, action_dim: int, embedding_dim: int, head_hidden_dim: int = 128) -> None:
        super(ValueAC, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._act_dim = squeeze(action_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2

        input_dim = embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_dim, head_hidden_dim))
            layers.append(self._act)
            input_dim = head_hidden_dim
        layers.append(nn.Linear(input_dim, self._act_dim))
        self._actor = nn.Sequential(*layers)

        input_dim = embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_dim, head_hidden_dim))
            input_dim = head_hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self._critic = nn.Sequential(*layers)

    def _setup_encoder(self) -> torch.nn.Module:
        raise NotImplementedError

    def _critic_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._critic(x).squeeze(1)

    def _actor_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._actor(x)

    def compute_action_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # for compatible, but we recommend use dict as input format
        if isinstance(inputs, torch.Tensor):
            embedding = self._encoder(inputs)
        else:
            embedding = self._encoder(inputs['obs'])
        value = self._critic_forward(embedding)
        logit = self._actor_forward(embedding)
        return {'value': value, 'logit': logit}

    def compute_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            embedding = self._encoder(inputs)
        else:
            embedding = self._encoder(inputs['obs'])
        logit = self._actor_forward(embedding)
        return {'logit': logit}


class ConvValueAC(ValueAC):

    def _setup_encoder(self) -> torch.nn.Module:
        return ConvEncoder(self._obs_dim, self._embedding_dim)

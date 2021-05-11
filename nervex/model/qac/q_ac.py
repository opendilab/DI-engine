import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional

from nervex.utils import squeeze, deep_merge_dicts, MODEL_REGISTRY
from nervex.torch_utils.network.nn_module import MLP
from ..common import ActorCriticBase, FCEncoder


class FCContinuousNet(nn.Module):
    """
    Overview:
        FC continuous network which is used in ``QAC``.
        A main feature is that it uses ``_use_final_tanh`` to control whether
        add a tanh layer to scale the output to (-1, 1).
    Interface:
        __init__, forward
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            embedding_dim: int = 64,
            use_final_tanh: bool = False,
            layer_num: int = 1,
    ) -> None:
        super(FCContinuousNet, self).__init__()
        self._act = nn.ReLU()
        self._main = nn.Sequential(
            MLP(input_dim, embedding_dim, embedding_dim, layer_num + 1, activation=self._act),
            nn.Linear(embedding_dim, output_dim)
        )
        self._use_final_tanh = use_final_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        if self._use_final_tanh:
            x = torch.tanh(x)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


@MODEL_REGISTRY.register('qac')
class QAC(ActorCriticBase):
    """
    Overview:
        QAC network. Use ``FCContinuousNet`` as subnetworks for actor and critic(s).
    Interface:
        __init__, forward, seed, optimize_actor, compute_q, compute_action, mimic
    """

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, tuple],
            state_action_embedding_dim: int = 64,
            state_embedding_dim: int = 64,
            twin_critic: bool = False,
    ) -> None:
        """
        Overview:
            Init actor network and critic network(s).
        Arguments:
            - obs_dim (:obj:`tuple`): tuple type observation dim
            - action_dim (:obj:`Union[int, tuple]`): int or tuple type action dim
            - state_action_embedding_dim (:obj:`int`): the dim of state + action that will be embedded into, \
                i.e. hidden dim
            - state_embedding_dim (:obj:`int`): the dim of state that will be embedded into, i.e. hidden dim
            - twin_critic (:obj:`bool`): whether to use a pair of critic networks. If True, it is TD3 model; \
                Otherwise, it is DDPG model.
        """
        super(QAC, self).__init__()

        self._act = nn.ReLU()
        # input info
        self._obs_dim: int = squeeze(obs_dim)
        self._act_dim: int = squeeze(action_dim)
        # embedding_dim
        self._state_action_embedding_dim = state_action_embedding_dim
        self._state_embedding_dim = state_embedding_dim
        # network
        self._twin_critic = twin_critic
        self._actor = FCContinuousNet(self._obs_dim, self._act_dim, self._state_embedding_dim, use_final_tanh=True)
        critic_num = 2 if twin_critic else 1
        self._critic = nn.ModuleList(
            [
                FCContinuousNet(self._obs_dim + self._act_dim, 1, self._state_action_embedding_dim)
                for _ in range(critic_num)
            ]
        )
        self._twin_critic = twin_critic

    def _critic_forward(self, x: torch.Tensor, single: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        if self._twin_critic and not single:
            return [self._critic[i](x) for i in range(2)]
        else:
            return self._critic[0](x)

    def _actor_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._actor(x)

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        action = inputs['action']
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        state_action_input = torch.cat([inputs['obs'], action], dim=1)
        q = self._critic_forward(state_action_input)
        return {'q_value': q}

    def compute_actor(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        action = self._actor_forward(obs)
        return {'action': action}

    @property
    def actor(self) -> torch.nn.Module:
        return self._actor

    @property
    def critic(self) -> torch.nn.Module:
        return self._critic

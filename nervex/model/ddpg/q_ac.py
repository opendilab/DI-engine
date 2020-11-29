import torch
import torch.nn as nn
from typing import Dict, List, Union

from ..common_arch import QActorCriticBase, FCEncoder
from nervex.utils import squeeze, deep_merge_dicts


class QAC(QActorCriticBase):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            action_range: dict,
            state_action_embedding_dim: int = 64,
            state_embedding_dim: int = 64,
            head_hidden_dim: int = 128,
            use_twin_critic: bool = False
    ) -> None:
        super(QAC, self).__init__()

        def backward_hook(module, grad_input, grad_output):
            for p in module.parameters():
                p.requires_grad = True

        self._act = nn.ReLU()
        # input info
        self._obs_dim: int = squeeze(obs_dim)
        self._act_dim: int = squeeze(action_dim)
        self._act_range = action_range
        # embedding_dim
        self._state_action_embedding_dim = state_action_embedding_dim
        self._state_embedding_dim = state_embedding_dim
        # encoder
        self._actor_encoder = self._setup_actor_encoder()
        self._critic_num = 2 if use_twin_critic else 1
        self._critic_encoder = nn.ModuleList([self._setup_critic_encoder() for _ in range(self._critic_num)])
        self._critic_encoder[0].register_backward_hook(backward_hook)
        # head
        self._head_layer_num = 2
        # actor head
        actor_input_dim = state_embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(actor_input_dim, head_hidden_dim))
            layers.append(self._act)
            actor_input_dim = head_hidden_dim
        layers.append(nn.Linear(actor_input_dim, self._act_dim))
        self._actor = nn.Sequential(*layers)
        # (twin) critic head
        self._critic = nn.ModuleList()
        for _ in range(self._critic_num):
            critic_input_dim = state_action_embedding_dim
            layers = []
            for _ in range(self._head_layer_num):
                layers.append(nn.Linear(critic_input_dim, head_hidden_dim))
                layers.append(self._act)
                critic_input_dim = head_hidden_dim
            layers.append(nn.Linear(critic_input_dim, 1))
            self._critic.append(nn.Sequential(*layers))
        self._critic[0].register_backward_hook(backward_hook)

    def _setup_critic_encoder(self) -> torch.nn.Module:
        raise NotImplementedError

    def _setup_actor_encoder(self) -> torch.nn.Module:
        raise NotImplementedError

    def _critic_forward(self, x: torch.Tensor, single: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        if single:
            ret = self._critic_encoder[0](x)
            ret = self._critic[0](ret).squeeze(1)
            return ret
        else:
            critic_ret = []
            for i in range(self._critic_num):
                ret = self._critic_encoder[i](x)
                ret = self._critic[i](ret).squeeze(1)
                critic_ret.append(ret)
            return critic_ret

    def _actor_forward(self, x: torch.Tensor) -> torch.Tensor:

        # def scale(num, new_min, new_max):
        #     return (num + 1) / 2 * (new_max - new_min) + new_min

        actor_ret = self._actor_encoder(x)
        actor_ret = self._actor(actor_ret)
        # actor_ret = scale(torch.tanh(actor_ret), self._act_range['min'], self._act_range['max'])
        return actor_ret.squeeze(1)

    def compute_action_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        q_dict = self.compute_q(inputs)
        action_dict = self.compute_action(inputs)
        return deep_merge_dicts(q_dict, action_dict)

    def compute_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        state_action_input = torch.cat([inputs['obs'], inputs['act']], dim=1)
        q = self._critic_forward(state_action_input)
        return {'q_value': q}

    def compute_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        action = self._actor_forward(state_input)
        return {'action': action}

    def optimize_actor(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        action = self._actor_forward(state_input)
        state_action_input = torch.cat([state_input, action], dim=1)

        for p in self._critic_encoder[0].parameters():
            p.requires_grad = False  # will set True when backward_hook called
        for p in self._critic[0].parameters():
            p.requires_grad = False  # will set True when backward_hook called
        q = self._critic_forward(state_action_input, single=True)

        return {'q_value': q}


class FCQAC(QAC):

    def _setup_critic_encoder(self) -> torch.nn.Module:
        return FCEncoder(self._obs_dim + self._act_dim, self._state_action_embedding_dim)

    def _setup_actor_encoder(self) -> torch.nn.Module:
        return FCEncoder(self._obs_dim, self._state_embedding_dim)

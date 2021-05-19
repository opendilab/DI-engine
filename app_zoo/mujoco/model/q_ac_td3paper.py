import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional

from nervex.model.common import QActorCriticBase
from nervex.utils import squeeze


class FCContinuousNet(nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: List[int] = [400, 300],
            use_final_tanh: bool = False,
    ) -> None:
        super(FCContinuousNet, self).__init__()
        self._act = nn.ReLU()
        self._use_final_tanh = use_final_tanh
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(self._act)
        if len(hidden_dim) > 1:
            for hid1, hid2 in zip(hidden_dim, hidden_dim[1:]):
                layers.append(nn.Linear(hid1, hid2))
                layers.append(self._act)
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self._main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        if self._use_final_tanh:
            x = torch.tanh(x)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


class QAC_Td3paper(QActorCriticBase):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, tuple],
            twin_critic: bool = False,
            use_backward_hook: bool = False,
    ) -> None:
        super(QAC_Td3paper, self).__init__()

        def backward_hook(module, grad_input, grad_output):
            for p in module.parameters():
                p.requires_grad = True

        self._act = nn.ReLU()
        # input info
        self._obs_dim: int = squeeze(obs_dim)
        self._act_dim: int = squeeze(action_dim)
        # hidden_dim
        # Original QAC in nervex/model has hidden layers with same hidden nodes,
        # while TD3 paper proposed 2 hidden layers with hidden nodes [400, 300].
        # That is why we implement a new version of QAC and FCContinuousNet here.
        self._hidden_dim = [400, 300]
        # network
        self._twin_critic = twin_critic
        self._actor = FCContinuousNet(self._obs_dim, self._act_dim, self._hidden_dim, use_final_tanh=True)
        critic_num = 2 if twin_critic else 1
        self._critic = nn.ModuleList(
            [FCContinuousNet(self._obs_dim + self._act_dim, 1, self._hidden_dim) for _ in range(critic_num)]
        )
        self._twin_critic = twin_critic
        self._use_backward_hook = use_backward_hook
        if self._use_backward_hook:
            self._critic[0].register_backward_hook(backward_hook)

    def _critic_forward(self, x: torch.Tensor, single: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        if self._twin_critic and not single:
            return [self._critic[i](x) for i in range(2)]
        else:
            return self._critic[0](x)

    def _collector_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._actor(x)

    def compute_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        action = inputs['action']
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        state_action_input = torch.cat([inputs['obs'], action], dim=1)
        q = self._critic_forward(state_action_input)
        return {'q_value': q}

    def compute_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        action = self._collector_forward(inputs['obs'])
        return {'action': action}

    def optimize_actor(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_input = inputs['obs']
        action = self._collector_forward(state_input)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        state_action_input = torch.cat([state_input, action], dim=1)

        if self._use_backward_hook:
            for p in self._critic[0].parameters():
                p.requires_grad = False  # will set True when backward_hook called
        q = self._critic_forward(state_action_input, single=True)

        return {'q_value': q}

    @property
    def collectorctor(self) -> torch.nn.Module:
        return self._actor

    @property
    def critic(self) -> torch.nn.Module:
        return self._critic

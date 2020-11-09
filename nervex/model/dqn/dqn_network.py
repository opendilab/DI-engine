from functools import partial
from typing import Union, List, Dict

import torch
import torch.nn as nn

from nervex.model import DuelingHead, ConvEncoder
from nervex.utils import squeeze


class FCDQN(nn.Module):

    def __init__(
        self, obs_dim, action_dim, hidden_dim_list=[128, 128, 128], dueling=True, a_layer_num=1, v_layer_num=1
    ):
        super(FCDQN, self).__init__()
        self.act = nn.ReLU()
        layers = []
        input_dim = squeeze(obs_dim)
        for dim in hidden_dim_list:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.act)
            input_dim = dim
        self.main = nn.Sequential(*layers)
        self.action_dim = squeeze(action_dim)
        self.dueling = dueling

        head_fn = partial(DuelingHead, a_layer_num=a_layer_num, v_layer_num=v_layer_num) if dueling else nn.Linear
        if isinstance(self.action_dim, list) or isinstance(self.action_dim, tuple):
            self.pred = nn.ModuleList()
            for dim in self.action_dim:
                self.pred.append(head_fn(input_dim, dim))
        else:
            self.pred = head_fn(input_dim, self.action_dim)

    def forward(self, x):
        x = self.main(x)
        if isinstance(self.action_dim, list):
            x = [m(x) for m in self.pred]
        else:
            x = self.pred(x)
        return {'logit': x}


class ConvDQN(nn.Module):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, list],
            head_hidden_dim: int = 64,
            dueling: bool = True,
            a_layer_num: int = 1,
            v_layer_num: int = 1
    ) -> None:
        super(ConvDQN, self).__init__()
        self.act = nn.ReLU()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = ConvEncoder(obs_dim, head_hidden_dim)

        head_fn = partial(DuelingHead, a_layer_num=a_layer_num, v_layer_num=v_layer_num) if dueling else nn.Linear
        if isinstance(self.action_dim, list) or isinstance(self.action_dim, tuple):
            self.pred = nn.ModuleList()
            for dim in self.action_dim:
                self.pred.append(head_fn(head_hidden_dim, dim))
        else:
            self.pred = head_fn(head_hidden_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        x = self.encoder(x)
        if isinstance(self.action_dim, list):
            x = [m(x) for m in self.pred]
        else:
            x = self.pred(x)
        return {'logit': x}

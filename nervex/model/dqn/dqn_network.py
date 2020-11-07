from functools import partial
from typing import Union, List

import torch
import torch.nn as nn

from nervex.model import DuelingHead
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
        return x


class ConvDQN(nn.Module):

    def __init__(
            self,
            obs_dim: tuple,
            action_dim: Union[int, list],
            dueling: bool = True,
            a_layer_num: int = 1,
            v_layer_num: int = 1
    ) -> None:
        super(ConvDQN, self).__init__()
        self.act = nn.ReLU()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        layers = []
        hidden_dim_list = [32, 64, 64]
        kernel_size = [8, 4, 3]
        stride = [4, 2, 1]
        input_dim = obs_dim[0]
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, hidden_dim_list[i], kernel_size[i], stride[i]))
            layers.append(self.act)
            input_dim = hidden_dim_list[i]
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)
        head_fn = partial(DuelingHead, a_layer_num=a_layer_num, v_layer_num=v_layer_num) if dueling else nn.Linear
        flatten_dim = self._get_flatten_dim()
        self.mid = nn.Linear(flatten_dim, input_dim)
        if isinstance(self.action_dim, list) or isinstance(self.action_dim, tuple):
            self.pred = nn.ModuleList()
            for dim in self.action_dim:
                self.pred.append(head_fn(input_dim, dim))
        else:
            self.pred = head_fn(input_dim, self.action_dim)

    def _get_flatten_dim(self) -> int:
        test_data = torch.randn(1, *self.obs_dim)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.main(x)
        x = self.mid(x)
        if isinstance(self.action_dim, list):
            x = [m(x) for m in self.pred]
        else:
            x = self.pred(x)
        return x

from typing import Union
import torch
import torch.nn as nn
from nervex.torch_utils import fc_block


class DuelingHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            action_dim: int,
            a_layer_num: int,
            v_layer_num: int,
            activation: Union[None, nn.Module] = nn.ReLU(),
            norm_type: Union[None, str] = None
    ) -> None:
        super(DuelingHead, self).__init__()
        self.A = [
            fc_block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type) for _ in range(a_layer_num)
        ]
        self.V = [
            fc_block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type) for _ in range(v_layer_num)
        ]

        self.A += fc_block(hidden_dim, action_dim, activation=None, norm_type=None)
        self.V += fc_block(hidden_dim, 1, activation=None, norm_type=None)

        self.A = nn.Sequential(*self.A)
        self.V = nn.Sequential(*self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.A(x)
        v = self.V(x)

        return a - a.mean(dim=-1, keepdim=True) + v

import torch
import torch.nn as nn
from functools import partial
from nervex.model import DuelingHead


class FCDQN(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim_list=[128, 256, 256],
        device='cpu',
        dueling=True,
        a_layer_num=1,
        v_layer_num=1
    ):
        super(FCDQN, self).__init__()
        self.act = nn.ReLU()
        layers = []
        for dim in hidden_dim_list:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.act)
            input_dim = dim
        self.main = nn.Sequential(*layers)
        self.action_dim = action_dim
        self.dueling = dueling

        head_fn = partial(DuelingHead, a_layer_num=a_layer_num, v_layer_num=v_layer_num) if dueling else nn.Linear
        if isinstance(self.action_dim, list):
            self.pred = nn.ModuleList()
            for dim in self.action_dim:
                self.pred.append(head_fn(input_dim, dim))
        else:
            self.pred = head_fn(input_dim, action_dim)
        self.device = device

    def forward(self, x, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = self.main(x)
        if isinstance(self.action_dim, list):
            x = [m(x) for m in self.pred]
        else:
            x = self.pred(x)
        return x

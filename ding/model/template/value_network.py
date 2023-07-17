import torch
from torch import nn
from ding.torch_utils import multilayer_perceptron


class QModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_num = cfg.model_num if hasattr(cfg, 'model_num') else 1
        self.models = nn.ModuleList([multilayer_perceptron(cfg.model) for _ in range(self.model_num)])

    def forward(self, obs, action):
        if self.model_num == 1:
            return self.models[0](torch.cat((obs, action), dim=1)).squeeze(dim=1)
        else:
            return torch.cat([model(torch.cat((obs, action), dim=1)) for model in self.models], dim=1)

    def min_q(self, obs, action):
        return torch.min(
            input=torch.cat([model(torch.cat((obs, action), dim=1)) for model in self.models], dim=1), dim=1
        ).values


class VModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_num = cfg.model_num if hasattr(cfg, 'model_num') else 1
        self.models = nn.ModuleList([multilayer_perceptron(cfg.model) for _ in range(self.model_num)])

    def forward(self, obs):
        if self.model_num == 1:
            return self.models[0](obs).squeeze(dim=1)
        else:
            return torch.cat([model(obs) for model in self.models], dim=1)

    def min_q(self, obs):
        return torch.min(input=torch.cat([model(obs) for model in self.models], dim=1), dim=1).values

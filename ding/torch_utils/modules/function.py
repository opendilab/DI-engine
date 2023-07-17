import torch
from torch import nn
from torch.distributions.transforms import TanhTransform
from .perceptron import multilayer_perceptron


class NonegativeFunction(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = multilayer_perceptron(cfg)

    def forward(self, x):
        return torch.exp(self.model(x))


class TanhFunction(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.transform = TanhTransform(cache_size=1)
        self.model = multilayer_perceptron(cfg)

    def forward(self, x):
        return self.transform(self.model(x))

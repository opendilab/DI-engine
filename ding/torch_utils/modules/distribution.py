import torch
from torch import nn


class Distribution(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, x, condition=None, **kwargs):
        raise NotImplementedError

    def sample(self, num=1, condition=None, **kwargs):
        with torch.no_grad():
            return self.rsample(num, condition, **kwargs)

    def rsample(self, num=1, condition=None, **kwargs):
        raise NotImplementedError

    def entropy(self, *args, **kwargs):
        raise NotImplementedError

    def dist(self, *args, **kwargs):
        raise NotImplementedError

    def sample_and_log_prob(self, num=1, condition=None, **kwargs):
        with torch.no_grad():
            return self.rsample_and_log_prob(num, condition, **kwargs)

    def rsample_and_log_prob(self, num=1, condition=None, **kwargs):
        raise NotImplementedError

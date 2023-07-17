import torch
from torch import nn
from ding.torch_utils import Gaussian, GaussianTanh


class StochasticPolicy(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.model_type == 'Gaussian':
            self.model = Gaussian(cfg.model)
        elif cfg.model_type == 'GaussianTanh':
            self.model = GaussianTanh(cfg.model)
        else:
            raise NotImplementedError

    def forward(self, obs):
        action, log_prob = self.model(obs)
        return action, log_prob

    def log_prob(self, action, obs):
        return self.model.log_prob(action, obs)

    def sample(self, obs, sample_shape=torch.Size()):
        return self.model.sample(obs, sample_shape)

    def rsample(self, obs, sample_shape=torch.Size()):
        return self.model.rsample(obs, sample_shape)

    def entropy(self, obs):
        return self.model.entropy(obs)

    def dist(self, obs):
        return self.model.dist(obs)

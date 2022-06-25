"""Adapted from https://github.com/kevinzakka/ibc. 
"""

from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from ding.utils import MODEL_REGISTRY, OPTIMIZER_REGISTRY


def create_optimizer(optimizer_config):
    return MODEL_REGISTRY.build(optimizer_config.pop("type"), **optimizer_config)

class StochasticOptimizer(ABC):
    @abstractmethod
    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        """Sample counter-negatives for feeding to the InfoNCE objective."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation."""
        raise NotImplementedError

@OPTIMIZER_REGISTRY.register('dfo')
class DFO(StochasticOptimizer):
    def __init__(
        self,
        bounds: list[list],
        noise_scale: float = 0.33,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 256,
        inference_samples: int = 2 ** 14,
    ):
        self.bounds = bounds
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        pass

    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        pass

@OPTIMIZER_REGISTRY.register('mcmc')
class MCMC(StochasticOptimizer):
    pass

@MODEL_REGISTRY.register('ebm')
class EBM(nn.Module):
    
    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        *args, # TODO: parameters
        optimizer_config: dict,
    ):
        self.net = nn.Linear(obs_shape + action_shape, 1)
        self.optimizer = create_optimizer(optimizer_config)

    def forward(self, obs, action=None):
        if not action:
            return self.optimizer.infer(obs, self)
        else:
            # obs: (B, Do)
            # action: (B, N+1, Da)
            energy = None
            return energy



@MODEL_REGISTRY.register('arebm')
class AutoregressiveEBM(nn.Module):

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        *args, # TODO: parameters
        optimizer_config: dict,
    ):
        self.net = nn.Linear(obs_shape + action_shape, 1)
        self.optimizer = create_optimizer(optimizer_config)

    def forward(self, obs, action=None):
        pass

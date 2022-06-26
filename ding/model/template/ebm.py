"""Adapted from https://github.com/kevinzakka/ibc. 
"""

from typing import Union, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from ding.utils import MODEL_REGISTRY, STOCHASTIC_OPTIMIZER_REGISTRY
from ding.torch_utils import unsqueeze_repeat
from ..common import RegressionHead


def create_stochastic_optimizer(stochastic_optimizer_config):
    return STOCHASTIC_OPTIMIZER_REGISTRY.build(
        stochastic_optimizer_config.pop("type"),
        **stochastic_optimizer_config
    )


class StochasticOptimizer(ABC):

    def set_action_bounds(self, action_bounds: np.ndarray):
        self.action_bounds = action_bounds

    @abstractmethod
    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        """Sample counter-negatives for feeding to the InfoNCE objective."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation."""
        raise NotImplementedError


@STOCHASTIC_OPTIMIZER_REGISTRY.register('dfo')
class DFO(StochasticOptimizer):
    def __init__(
        self,
        # action_bounds: np.ndarray,
        noise_scale: float = 0.1,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 256,
        inference_samples: int = 512,
        cuda: bool = False,
    ):
        # set later by `set_action_bounds`
        self.action_bounds = None
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.device = torch.device('cuda' if cuda else "cpu")

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self.action_bounds.shape[1])
        samples = np.random.uniform(self.action_bounds[0, :], self.action_bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples, -1)

    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        noise_scale = self.noise_scale
        action_bounds = torch.as_tensor(self.action_bounds).to(self.device)

        action_samples = self._sample(obs.size(0) * self.inference_samples)
        # (B, N, A)
        action_samples = action_samples.reshape(obs.size(0), self.inference_samples, -1)

        for i in range(self.iters):
            # Compute energies.
            # obs: (B, O)
            # action_samples: (B, N, A)
            # energy: (B, N)
            energies = ebm.forward(obs, action_samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            action_samples = action_samples[torch.arange(action_samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            action_samples = action_samples + torch.randn_like(action_samples) * noise_scale
            action_samples = action_samples.clamp(min=action_bounds[0, :], max=action_bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm.forward(obs, action_samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        # (B, )
        best_idxs = probs.argmax(dim=-1)
        return action_samples[torch.arange(action_samples.size(0)), best_idxs, :]


@STOCHASTIC_OPTIMIZER_REGISTRY.register('mcmc')
class MCMC(StochasticOptimizer):
    pass


@MODEL_REGISTRY.register('ebm')
class EBM(nn.Module):
    
    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size: int = 64,
        hidden_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        input_size = obs_shape + action_shape 
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            activation,
            RegressionHead(
                hidden_size,
                1,
                hidden_layer_num,
                final_tanh=False,
                activation=activation,
                norm_type=norm_type
            )
        )

    def forward(self, obs, action):
        # obs: (B, O)
        # action: (B, N, A)
        obs = unsqueeze_repeat(obs, action.shape[1], 1)
        x = torch.concat([obs, action], -1)
        x = self.net(x)
        return x['pred']


@MODEL_REGISTRY.register('arebm')
class AutoregressiveEBM(nn.Module):

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        **kwargs,
    ):
        super().__init__()

    def forward(self, obs, action):
        # obs: (B, O)
        # action: (B, N, A)
        pass

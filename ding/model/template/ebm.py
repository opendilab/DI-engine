"""Vanilla DFO and EBM are adapted from https://github.com/kevinzakka/ibc. 
"""

from dataclasses import replace
from typing import Union, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from ding.utils import MODEL_REGISTRY, STOCHASTIC_OPTIMIZER_REGISTRY
from ding.torch_utils import unsqueeze_repeat, fold_batch, unfold_batch
from ding.torch_utils.network.gtrxl import PositionalEmbedding
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

    @torch.no_grad()
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


@STOCHASTIC_OPTIMIZER_REGISTRY.register('ardfo')
class AutoRegressiveDFO(DFO):
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
        super().__init__(noise_scale, noise_shrink, iters, train_samples, inference_samples, cuda)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        noise_scale = self.noise_scale
        action_bounds = torch.as_tensor(self.action_bounds).to(self.device)

        action_samples = self._sample(obs.size(0) * self.inference_samples)
        # (B, N, A)
        action_samples = action_samples.reshape(obs.size(0), self.inference_samples, -1)

        for i in range(self.iters):

            # (B, N, A)
            energies = ebm.forward(obs, action_samples)
            probs = F.softmax(-1 * energies, dim=1)

            for j in range(energies.shape[-1]):
                # TODO: move `energies = ebm.forward(obs, action_samples)` into inner loop?
                _action_samples = action_samples[:, :, j]
                _probs = probs[:, :, j]
                _idxs = torch.multinomial(_probs, self.inference_samples, replacement=True)
                _action_samples = _action_samples[torch.arange(_action_samples.size(0)).unsqueeze(-1), _idxs]

                _action_samples = _action_samples + torch.randn_like(_action_samples) * noise_scale
                _action_samples = _action_samples.clamp(min=action_bounds[0, j], max=action_bounds[1, j])

                action_samples[:, :, j] = _action_samples

            noise_scale *= self.noise_shrink
        
        # (B, N, A)
        energies = ebm.forward(obs, action_samples)
        probs = F.softmax(-1 * energies, dim=1)
        # (B)
        best_idxs = probs[:, :, -1].argmax(dim=1)
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
        **kwargs,
    ):
        super().__init__()
        input_size = obs_shape + action_shape 
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(),
            RegressionHead(
                hidden_size,
                1,
                hidden_layer_num,
                final_tanh=False,
            )
        )

    def forward(self, obs, action):
        # obs: (B, O)
        # action: (B, N, A)
        # return: (B, N)
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
        d_model: int = 64,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        cuda: bool = False,
        **kwargs,
    ):
        # treat obs_dim, and action_dim as sequence_dim
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = torch.device('cuda' if cuda else "cpu")
        self.obs_embed_layer = nn.Linear(1, d_model)
        self.action_embed_layer = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.action_mask = self.transformer.generate_square_subsequent_mask(action_shape).to(self.device)
        self.output_layer = nn.Linear(d_model, 1)
        self._generate_positional_encoding(d_model)

    def _generate_positional_encoding(self, d_model):
        positional_encoding_layer = PositionalEmbedding(d_model)
        # batch_first
        self.obs_pe = positional_encoding_layer(
            PositionalEmbedding.generate_pos_seq(self.obs_shape)
        ).permute(1, 0, 2).contiguous().to(self.device)
        self.action_pe = positional_encoding_layer(
            PositionalEmbedding.generate_pos_seq(self.action_shape)
        ).permute(1, 0, 2).contiguous().to(self.device)

    def forward(self, obs, action):
        # obs: (B, O)
        # action: (B, N, A)
        # return: (B, N, A)
        obs = unsqueeze_repeat(obs, action.shape[1], 1)
        # obs: (B*N, O)
        # action: (B*N, A)
        obs, batch_dims = fold_batch(obs)
        action, _ = fold_batch(action)

        # obs: (B*N, O, 1)
        # action: (B*N, A, 1)
        # the second dimension (O, A) is now interpreted as sequence dimension
        # so that `obs`, `action` can be used as `src` and `tgt` to `nn.Transformer` 
        # block with `batch_first=False`
        obs = self.obs_embed_layer(obs.unsqueeze(-1)) + self.obs_pe.to(obs.device)
        action = self.action_embed_layer(action.unsqueeze(-1)) + self.action_pe.to(obs.device)

        output = self.transformer(src=obs, tgt=action, tgt_mask=self.action_mask)

        # output: (B*N, A)
        output = self.output_layer(output).squeeze(-1)

        # output(energy): (B, N, A)
        output = unfold_batch(output, batch_dims)

        return output

import torch
import treetensor.torch as ttorch
from torch.distributions import Normal, Independent


class ArgmaxSampler:

    def __call__(self, logit: torch.Tensor) -> torch.Tensor:
        return logit.argmax(dim=-1)


class MultinomialSampler:

    def __call__(self, logit: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=logit)
        return dist.sample()


class MuSampler:

    def __call__(self, logit: ttorch.Tensor) -> torch.Tensor:
        return logit.mu


class ReparameterizationSampler:

    def __call__(self, logit: ttorch.Tensor) -> torch.Tensor:
        dist = Normal(logit.mu, logit.sigma)
        dist = Independent(dist, 1)
        return dist.rsample()

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


class HybridStochasticSampler:

    def __call__(self, logit: ttorch.Tensor) -> ttorch.Tensor:
        dist = torch.distributions.Categorical(logits=logit.action_type)
        action_type = dist.sample()
        dist = Normal(logit.action_args.mu, logit.action_args.sigma)
        dist = Independent(dist, 1)
        action_args = dist.rsample()
        return ttorch.as_tensor({
            'action_type': action_type,
            'action_args': action_args,
        })


class HybridDeterminsticSampler:

    def __call__(self, logit: ttorch.Tensor) -> ttorch.Tensor:
        action_type = logit.action_type.argmax(dim=-1)
        action_args = logit.action_args.mu
        return ttorch.as_tensor({
            'action_type': action_type,
            'action_args': action_args,
        })

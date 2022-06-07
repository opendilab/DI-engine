import pytest
import torch
from torch.distributions import Normal, Independent, TransformedDistribution, TanhTransform
from ding.policy.mbpolicy import MBSACPolicy


@pytest.mark.unittest
def test_TanhTransform():
    mu, sigma = torch.randn(3, 3), torch.ones(3, 3)

    dist = Independent(Normal(mu, sigma), 1)
    pred = dist.rsample()
    action = torch.tanh(pred)

    log_prob_1 = dist.log_prob(pred
                               ) + 2 * (pred + torch.nn.functional.softplus(-2. * pred) -
                                        torch.log(torch.tensor(2.))).sum(-1)

    tanh_dist = TransformedDistribution(Independent(Normal(mu, sigma), 1), [TanhTransform()])

    log_prob_2 = tanh_dist.log_prob(action)

    assert (log_prob_1 - log_prob_2).mean().abs() < 1e-5

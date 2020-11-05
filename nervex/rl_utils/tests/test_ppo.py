import pytest
from itertools import product
import torch
from nervex.rl_utils import ppo_data, ppo_error

use_value_clip_args = [True, False]
dual_clip_args = [None, 5.0]
args = [item for item in product(*[use_value_clip_args, dual_clip_args])]


@pytest.mark.unittest
@pytest.mark.parametrize('use_value_clip, dual_clip', args)
def test_ppo(use_value_clip, dual_clip):
    B = 4
    logp_new = torch.randn(B)
    logp_old = torch.randn(B) + torch.rand_like(logp_new) * 0.1
    value_new = torch.randn(B)
    value_old = torch.randn(B) + torch.rand_like(value_new) * 0.1
    adv = torch.rand(B) * 0.5
    return_ = torch.randn(B) * 2
    data = ppo_data(logp_new, logp_old, value_new, value_old, adv, return_)
    loss, info = ppo_error(data, use_value_clip=use_value_clip, dual_clip=dual_clip)

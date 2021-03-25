import sys
import os

import torch
import numpy as np
from itertools import product
import pytest

from nervex.model import FCPPG
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

obs_dim = [
    4, 8
]
action_dim = [2, 6]

args = [item for item in product(*[obs_dim, action_dim])]


def output_check(action_dim, model, output):
    if isinstance(action_dim, tuple):
        loss = sum([t.sum() for t in output])
    elif np.isscalar(action_dim):
        loss = output.sum()
    is_differentiable(loss, model)


@pytest.mark.unittest
@pytest.mark.parametrize('obs_dim, action_dim', args)
def test_fc_ppg(obs_dim, action_dim):
    input = {'obs': torch.randn(4, obs_dim), 'action': torch.randn(4, squeeze(action_dim))}
    model = FCPPG(obs_dim, action_dim)
    # compute_action_value
    outputs = model(input, mode='compute_action_value')
    value, logit = outputs['value'], outputs['logit']
    output_check(model._act_dim, [model._policy_net._encoder, model._policy_net._critic], value)

    # compute_value
    value = model(input, mode='compute_value')['value']
    print("value: ", value)
    assert value.shape == (4, )
    output_check(model._act_dim, [model._value_net._encoder, model._value_net._critic], value)

    for p in model.parameters():
        p.grad = None
    # compute_action
    logit = model(input, mode='compute_action')['logit']
    assert logit.shape == (4, squeeze(action_dim))
    assert logit.eq(logit.clamp(-1, 1)).all()
    output_check(model._act_dim, [model._policy_net._encoder, model._policy_net._actor], logit)


import sys
import os

import torch
import numpy as np
from itertools import product
import pytest

from nervex.model import FCPPG
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

obs_shape = [4, 8]
action_shape = [2, 6]

args = [item for item in product(*[obs_shape, action_shape])]


def output_check(action_shape, model, output):
    if isinstance(action_shape, tuple):
        loss = sum([t.sum() for t in output])
    elif np.isscalar(action_shape):
        loss = output.sum()
    is_differentiable(loss, model)


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, action_shape', args)
def test_fc_ppg(obs_shape, action_shape):
    input = torch.randn(4, obs_shape)
    model = FCPPG(obs_shape, action_shape)
    # compute_action_value
    outputs = model(input, mode='compute_actor_critic')
    value, logit = outputs['value'], outputs['logit']
    output_check(model._act_shape, [model._policy_net._encoder, model._policy_net._critic], value)

    # compute_value
    value = model(input, mode='compute_critic')['value']
    print("value: ", value)
    assert value.shape == (4, )
    output_check(model._act_shape, [model._value_net._encoder, model._value_net._critic], value)

    for p in model.parameters():
        p.grad = None
    # compute_action
    logit = model(input, mode='compute_actor')['logit']
    assert logit.shape == (4, squeeze(action_shape))
    assert logit.eq(logit.clamp(-1, 1)).all()
    output_check(model._act_shape, [model._policy_net._encoder, model._policy_net._actor], logit)

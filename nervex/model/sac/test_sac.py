import sys
import os

import torch
import numpy as np
from itertools import product
import pytest

from nervex.model import SAC
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

obs_shape = [
    32,
]
action_shape = [6, 1]

args = [item for item in product(*[obs_shape, action_shape])]


def output_check(action_shape, model, output):
    if isinstance(action_shape, tuple):
        loss = sum([t.sum() for t in output])
    elif np.isscalar(action_shape):
        loss = output.sum()
    is_differentiable(loss, model)


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, action_shape', args)
def test_sac(obs_shape, action_shape):
    input = {'obs': torch.randn(4, obs_shape), 'action': torch.randn(4, squeeze(action_shape))}
    model = SAC(obs_shape, action_shape, twin_q=False, value_network=True)
    # compute_q
    q_value = model(input, mode='compute_critic', qv='q')['q_value']
    print("q_value: ", q_value)
    assert q_value.shape == (4, )
    output_check(model._act_shape, model._soft_q_net, q_value)

    # compute_value
    v_value = model(input['obs'], mode='compute_critic', qv='v')['v_value']
    print("v_value: ", v_value)
    assert v_value.shape == (4, )
    output_check(model._act_shape, model._value_net, v_value)

    # compute_action
    action = model(input['obs'], mode='compute_actor')['action']
    assert action.shape == (4, squeeze(action_shape))
    assert action.eq(action.clamp(-1, 1)).all()
    print("action: ", action)

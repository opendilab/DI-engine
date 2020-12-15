import sys
import os

import torch
import numpy as np
from itertools import product
import pytest

from nervex.model import SAC
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

obs_dim = [32, ]
action_dim = [6, ]

input = [{'obs': torch.randn(4, 32), 'action': torch.randn(4, squeeze(action_dim))}]
args = [item for item in product(*[input, obs_dim, action_dim])]


def output_check(action_dim, model, output):
    if isinstance(action_dim, tuple):
        loss = sum([t.sum() for t in output])
    elif np.isscalar(action_dim):
        loss = output.sum()
    is_differentiable(loss, model)


@pytest.mark.unittest
@pytest.mark.parametrize('input, obs_dim, action_dim', args)
def test_sac(input, obs_dim, action_dim):
    model = SAC(obs_dim, action_dim, use_twin_q=False)
    # compute_q
    q_value = model(input, mode='compute_q')['q_value']
    print("q_value: ", q_value)
    assert q_value.shape == (4, )
    output_check(model._act_dim, model._soft_q_net, q_value)

    # compute_value
    v_value = model(input, mode='compute_value')['v_value']
    print("v_value: ", v_value)
    assert v_value.shape == (4, )
    output_check(model._act_dim, model._value_net, v_value)

    # evaluate
    eval_data = model(input, mode='evaluate')
    print("evaluate: ", eval_data)
    for k, v in eval_data.items():
        print(k, v.shape)

    # compute_action
    action = model(input, mode='compute_action')['action']
    if squeeze(action_dim) == 1:
        assert action.shape == (4, )
    else:
        assert action.shape == (4, squeeze(action_dim))
    assert action.eq(action.clamp(-1, 1)).all()
    print("action: ", action)

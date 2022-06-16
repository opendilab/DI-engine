import pytest
import numpy as np
import torch
from itertools import product

from ding.model import VAC
from ding.torch_utils import is_differentiable

from ding.model import ConvEncoder

B, C, H, W = 4, 3, 128, 128
obs_shape = [4, (8, ), (4, 64, 64)]
act_args = [[6, 'discrete'], [(3, ), 'continuous'], [[2, 3, 6], 'discrete']]
# act_args = [[(3, ), True]]
args = list(product(*[obs_shape, act_args, [False, True]]))


def output_check(model, outputs, action_shape):
    if isinstance(action_shape, tuple):
        loss = sum([t.sum() for t in outputs])
    elif np.isscalar(action_shape):
        loss = outputs.sum()
    is_differentiable(loss, model)


def model_check(model, inputs):
    outputs = model(inputs, mode='compute_actor_critic')
    value, logit = outputs['value'], outputs['logit']
    if model.action_space == 'continuous':
        outputs = value.sum() + logit['mu'].sum() + logit['sigma'].sum()
    else:
        if model.multi_head:
            outputs = value.sum() + sum([t.sum() for t in logit])
        else:
            outputs = value.sum() + logit.sum()
    output_check(model, outputs, 1)

    for p in model.parameters():
        p.grad = None
    logit = model(inputs, mode='compute_actor')['logit']
    if model.action_space == 'continuous':
        logit = logit['mu'].sum() + logit['sigma'].sum()
    output_check(model.actor, logit, model.action_shape)

    for p in model.parameters():
        p.grad = None
    value = model(inputs, mode='compute_critic')['value']
    assert value.shape == (B, )
    output_check(model.critic, value, 1)


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, act_args, share_encoder', args)
class TestVACGeneral:

    def test_vac(self, obs_shape, act_args, share_encoder):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = VAC(obs_shape, action_shape=act_args[0], action_space=act_args[1], share_encoder=share_encoder)
        model_check(model, inputs)


@pytest.mark.unittest
@pytest.mark.parametrize('share_encoder', [(False, ), (True, )])
class TestVACEncoder:

    def test_vac_with_impala_encoder(self, share_encoder):
        inputs = torch.randn(B, 4, 64, 64)
        model = VAC(
            obs_shape=(4, 64, 64),
            action_shape=6,
            action_space='discrete',
            share_encoder=share_encoder,
            impala_cnn_encoder=True
        )
        model_check(model, inputs)

    def test_encoder_assignment(self, share_encoder):
        inputs = torch.randn(B, 4, 64, 64)

        special_encoder = ConvEncoder(obs_shape=(4, 64, 64), hidden_size_list=[16, 32, 32, 64])

        model = VAC(
            obs_shape=(4, 64, 64),
            action_shape=6,
            action_space='discrete',
            share_encoder=share_encoder,
            actor_head_hidden_size=64,
            critic_head_hidden_size=64,
            encoder=special_encoder
        )
        model_check(model, inputs)

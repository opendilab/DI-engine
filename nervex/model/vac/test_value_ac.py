import torch
import numpy as np
import pytest

from nervex.model import ConvValueAC
from nervex.torch_utils import is_differentiable

B = 4
T = 6
embedding_size = 32
action_shape_args = [(6, ), [
    12,
]]


@pytest.mark.unittest
@pytest.mark.parametrize('action_shape', action_shape_args)
class TestValueAC:

    def output_check(self, model, outputs, action_shape):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_conv_value_ac(self, action_shape):
        C, H, W = 3, 128, 128
        inputs = {'obs': torch.randn(B, C, H, W)}
        model = ConvValueAC((C, H, W), action_shape, embedding_size)

        outputs = model(inputs, mode='compute_actor_critic')
        value, logit = outputs['value'], outputs['logit']
        self.output_check([model._encoder, model._critic], value, model._act_shape)

        for p in model.parameters():
            p.grad = None
        logit = model(inputs, mode='compute_actor')['logit']
        self.output_check([model._encoder, model._actor], logit, model._act_shape)

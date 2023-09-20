import torch
import pytest
from itertools import product

from ding.model.template import ACER
from ding.torch_utils import is_differentiable

B = 4
obs_shape = [4, (8, ), (4, 64, 64)]
act_shape = [3, (6, )]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
class TestACER:

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_ACER(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = ACER(obs_shape, act_shape)

        outputs_c = model(inputs, mode='compute_critic')
        assert isinstance(outputs_c, dict)
        if isinstance(act_shape, int):
            assert outputs_c['q_value'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs_c['q_value'].shape == (B, *act_shape)

        outputs_a = model(inputs, mode='compute_actor')
        assert isinstance(outputs_a, dict)
        if isinstance(act_shape, int):
            assert outputs_a['logit'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs_a['logit'].shape == (B, *act_shape)

        outputs = {**outputs_a, **outputs_c}
        loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

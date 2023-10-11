import torch
import pytest
from itertools import product

from ding.model.template import EDAC
from ding.torch_utils import is_differentiable

B = 4
obs_shape = [4, (8, )]
act_shape = [3, (6, )]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
class TestEDAC:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_EDAC(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs_obs = torch.randn(B, obs_shape)
        else:
            inputs_obs = torch.randn(B, *obs_shape)
        if isinstance(act_shape, int):
            inputs_act = torch.randn(B, act_shape)
        else:
            inputs_act = torch.randn(B, *act_shape)
        inputs = {'obs': inputs_obs, 'action': inputs_act}
        model = EDAC(obs_shape, act_shape, ensemble_num=2)

        outputs_c = model(inputs, mode='compute_critic')
        assert isinstance(outputs_c, dict)
        assert outputs_c['q_value'].shape == (2, B)
        self.output_check(model.critic, outputs_c)

        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        outputs_a = model(inputs, mode='compute_actor')
        assert isinstance(outputs_a, dict)
        if isinstance(act_shape, int):
            assert outputs_a['logit'][0].shape == (B, act_shape)
            assert outputs_a['logit'][1].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs_a['logit'][0].shape == (B, *act_shape)
            assert outputs_a['logit'][1].shape == (B, *act_shape)
        outputs = {'mu': outputs_a['logit'][0], 'sigma': outputs_a['logit'][1]}
        self.output_check(model.actor, outputs)

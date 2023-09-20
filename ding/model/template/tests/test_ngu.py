import pytest
from itertools import product
import torch
from ding.model.template import NGU
from ding.torch_utils import is_differentiable

B = 4
H = 4
obs_shape = [4, (8, ), (4, 64, 64)]
act_shape = [4, (4, )]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
class TestNGU:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_ngu(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs_obs = torch.randn(B, H, obs_shape)
        else:
            inputs_obs = torch.randn(B, H, *obs_shape)
        if isinstance(act_shape, int):
            inputs_prev_action = torch.ones(B, act_shape).long()
        else:
            inputs_prev_action = torch.ones(B, *act_shape).long()
        inputs_prev_reward_extrinsic = torch.randn(B, H, 1)
        inputs_beta = 2 * torch.ones([4, 4], dtype=torch.long)
        inputs = {
            'obs': inputs_obs,
            'prev_state': None,
            'prev_action': inputs_prev_action,
            'prev_reward_extrinsic': inputs_prev_reward_extrinsic,
            'beta': inputs_beta
        }

        model = NGU(obs_shape, act_shape, collector_env_num=3)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape, *act_shape)
        self.output_check(model, outputs['logit'])

        inputs = {
            'obs': inputs_obs,
            'prev_state': None,
            'action': inputs_prev_action,
            'reward': inputs_prev_reward_extrinsic,
            'prev_reward_extrinsic': inputs_prev_reward_extrinsic,
            'beta': inputs_beta
        }
        model = NGU(obs_shape, act_shape, collector_env_num=3)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape, *act_shape)
        self.output_check(model, outputs['logit'])

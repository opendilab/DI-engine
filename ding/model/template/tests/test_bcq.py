import pytest
from itertools import product
import torch
from ding.model.template import BCQ
from ding.torch_utils import is_differentiable

B = 4
obs_shape = [4, (8, )]
act_shape = [3, (6, )]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
class TestBCQ:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_BCQ(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs_obs = torch.randn(B, obs_shape)
        else:
            inputs_obs = torch.randn(B, *obs_shape)
        if isinstance(act_shape, int):
            inputs_act = torch.randn(B, act_shape)
        else:
            inputs_act = torch.randn(B, *act_shape)
        inputs = {'obs': inputs_obs, 'action': inputs_act}
        model = BCQ(obs_shape, act_shape)

        outputs_c = model(inputs, mode='compute_critic')
        assert isinstance(outputs_c, dict)
        if isinstance(act_shape, int):
            assert torch.stack(outputs_c['q_value']).shape == (2, B)
        else:
            assert torch.stack(outputs_c['q_value']).shape == (2, B)
        self.output_check(model.critic, torch.stack(outputs_c['q_value']))

        outputs_a = model(inputs, mode='compute_actor')
        assert isinstance(outputs_a, dict)
        if isinstance(act_shape, int):
            assert outputs_a['action'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs_a['action'].shape == (B, *act_shape)
        self.output_check(model.actor, outputs_a)

        outputs_vae = model(inputs, mode='compute_vae')
        assert isinstance(outputs_vae, dict)
        if isinstance(act_shape, int):
            assert outputs_vae['recons_action'].shape == (B, act_shape)
            assert outputs_vae['mu'].shape == (B, act_shape * 2)
            assert outputs_vae['log_var'].shape == (B, act_shape * 2)
            assert outputs_vae['z'].shape == (B, act_shape * 2)
        elif len(act_shape) == 1:
            assert outputs_vae['recons_action'].shape == (B, *act_shape)
            assert outputs_vae['mu'].shape == (B, act_shape[0] * 2)
            assert outputs_vae['log_var'].shape == (B, act_shape[0] * 2)
            assert outputs_vae['z'].shape == (B, act_shape[0] * 2)
        if isinstance(obs_shape, int):
            assert outputs_vae['prediction_residual'].shape == (B, obs_shape)
        else:
            assert outputs_vae['prediction_residual'].shape == (B, *obs_shape)

        outputs_eval = model(inputs, mode='compute_eval')
        assert isinstance(outputs_eval, dict)
        assert isinstance(outputs_eval, dict)
        if isinstance(act_shape, int):
            assert outputs_eval['action'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs_eval['action'].shape == (B, *act_shape)

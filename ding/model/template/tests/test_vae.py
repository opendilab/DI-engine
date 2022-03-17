import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template.vae import VanillaVAE


@pytest.mark.unittest
def test_vae():
    batch_size = 32
    action_shape = 6
    original_action_shape = 2
    obs_shape = 6
    hidden_size_list = [256, 256]
    inputs = {
        'action': torch.randn(batch_size, original_action_shape),
        'obs': torch.randn(batch_size, obs_shape),
        'next_obs': torch.randn(batch_size, obs_shape)
    }

    vae_model = VanillaVAE(original_action_shape, obs_shape, action_shape, hidden_size_list)
    outputs = vae_model(inputs)

    assert outputs['recons_action'].shape == (batch_size, original_action_shape)
    assert outputs['prediction_residual'].shape == (batch_size, obs_shape)
    assert isinstance(outputs['input'], dict)
    assert outputs['mu'].shape == (batch_size, obs_shape)
    assert outputs['log_var'].shape == (batch_size, obs_shape)
    assert outputs['z'].shape == (batch_size, action_shape)

    outputs_decode = vae_model.decode_with_obs(outputs['z'], inputs['obs'])
    assert outputs_decode['reconstruction_action'].shape == (batch_size, original_action_shape)
    assert outputs_decode['predition_residual'].shape == (batch_size, obs_shape)

    outputs['original_action'] = inputs['action']
    outputs['true_residual'] = inputs['next_obs'] - inputs['obs']
    vae_loss = vae_model.loss_function(outputs, kld_weight=0.01, predict_weight=0.01)
    is_differentiable(vae_loss['loss'], vae_model)

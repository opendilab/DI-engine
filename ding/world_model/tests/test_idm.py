import torch
import pytest
from itertools import product

from ding.world_model.idm import InverseDynamicsModel
from ding.torch_utils import is_differentiable
from ding.utils import squeeze

B = 4
obs_shape = 12
encoder_hidden_size_list = [10, 20, 10]
action_shape_args = [6, (6, ), [6]]
args = list(product(*[action_shape_args, ['regression', 'reparameterization']]))


@pytest.mark.unittest
@pytest.mark.parametrize('action_shape, action_space', args)
class TestContinousIDM:

    def test_continuous_idm(self, action_shape, action_space):

        model = InverseDynamicsModel(
            obs_shape=obs_shape,
            action_shape=action_shape,
            encoder_hidden_size_list=encoder_hidden_size_list,
            action_space=action_space,
        )
        inputs = {'obs': torch.randn(B, obs_shape * 2), 'action': torch.randn(B, squeeze(action_shape))}
        if action_space == 'regression':
            action = model.predict_action(inputs['obs'])['action']
            if isinstance(action_shape, int):
                assert action.shape == (B, action_shape)
            else:
                assert action.shape == (B, squeeze(action_shape))
            assert action.eq(action.clamp(-1, 1)).all()
        elif action_space == 'reparameterization':
            (mu, sigma) = model.predict_action(inputs['obs'])['logit']
            action = model.predict_action(inputs['obs'])['action']
            assert mu.shape == (B, action_shape)
            assert sigma.shape == (B, action_shape)
            assert action.shape == (B, action_shape)
        self.test_train(model, inputs)

    def test_train(self, model, inputs):
        loss = model.train(inputs, n_epoch=10, learning_rate=0.01, weight_decay=1e-4)
        is_differentiable(loss, model)


B = 4
obs_shape = [4, (8, ), (4, 64, 64)]
action_shape = [3, (6, ), [2, 3, 6]]
encoder_hidden_size_list = [10, 20, 10]
args = list(product(*[obs_shape, action_shape]))
action_space = 'discrete'


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, action_shape', args)
class TestDiscreteIDM:

    def test_train(self, model, inputs):
        loss = model.train(inputs, n_epoch=10, learning_rate=0.01, weight_decay=1e-4)
        is_differentiable(loss, model)

    def test_discrete_idm(self, obs_shape, action_shape):
        if isinstance(obs_shape, int):
            inputs = {'obs': torch.randn(B, obs_shape * 2), 'action': torch.randn(B, squeeze(action_shape))}
        else:
            obs_shape[0] *= 2
            inputs = torch.randn(B, *obs_shape)
        model = InverseDynamicsModel(
            obs_shape=obs_shape,
            action_shape=action_shape,
            encoder_hidden_size_list=encoder_hidden_size_list,
            action_space=action_space,
        )
        outputs = model.forward(inputs)
        assert isinstance(outputs, dict)
        if isinstance(action_shape, int):
            assert outputs['logit'].shape == (B, action_shape)
        else:
            assert outputs['logit'].shape == (B, *action_shape)
        self.test_train(model, inputs)
        action = model.predict_action(inputs)['action']
        assert action.shape == outputs['logit'].shape
        model.train(inputs, n_epoch=10, learning_rate=0.01, weight_decay=1e-4)

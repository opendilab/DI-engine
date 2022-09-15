import torch
import pytest
from itertools import product

from ding.world_model.idm import InverseDynamicsModel
from ding.torch_utils import is_differentiable
from ding.utils import squeeze

B = 4
obs_shape_arg = [4, (8, ), (9, 64, 64)]
encoder_hidden_size_list = [10, 20, 10]
action_shape_arg = [6, (6, ), [6]]
args = list(product(*[obs_shape_arg, action_shape_arg, ['regression', 'reparameterization']]))


@pytest.mark.unittest
class TestContinousIDM:

    @pytest.mark.parametrize('obs_shape, action_shape, action_space', args)
    def test_continuous_idm(self, obs_shape, action_shape, action_space):

        model = InverseDynamicsModel(
            obs_shape=obs_shape,
            action_shape=action_shape,
            encoder_hidden_size_list=encoder_hidden_size_list,
            action_space=action_space,
        )
        inputs = {}
        if isinstance(obs_shape, int):
            inputs['obs'] = torch.randn(B, obs_shape * 2)
        else:
            inputs['obs'] = torch.randn(B, *(obs_shape[0] * 2, *obs_shape[1:]))
        if isinstance(action_shape, int):
            inputs['action'] = torch.randn(B, action_shape)
        else:
            inputs['action'] = torch.randn(B, *action_shape)
        if action_space == 'regression':
            action = model.predict_action(inputs['obs'])['action']
            if isinstance(action_shape, int):
                assert action.shape == (B, action_shape)
            else:
                assert action.shape == (B, *action_shape)
            assert action.eq(action.clamp(-1, 1)).all()
        elif action_space == 'reparameterization':
            (mu, sigma) = model.predict_action(inputs['obs'])['logit']
            action = model.predict_action(inputs['obs'])['action']
            if isinstance(action_shape, int):
                assert mu.shape == (B, action_shape)
                assert sigma.shape == (B, action_shape)
                assert action.shape == (B, action_shape)
            else:
                assert mu.shape == (B, *action_shape)
                assert sigma.shape == (B, *action_shape)
                assert action.shape == (B, *action_shape)

        loss = model.train(inputs, n_epoch=10, learning_rate=0.01, weight_decay=1e-4)
        assert isinstance(loss, float)


B = 4
obs_shape = [4, (8, ), (4, 64, 64)]
action_shape = [6, (6, ), [6]]
encoder_hidden_size_list = [10, 20, 10]
args = list(product(*[obs_shape, action_shape]))
action_space = 'discrete'


@pytest.mark.unittest
class TestDiscreteIDM:

    @pytest.mark.parametrize('obs_shape, action_shape', args)
    def test_discrete_idm(self, obs_shape, action_shape):
        model = InverseDynamicsModel(
            obs_shape=obs_shape,
            action_shape=action_shape,
            encoder_hidden_size_list=encoder_hidden_size_list,
            action_space=action_space,
        )
        inputs = {}
        if isinstance(obs_shape, int):
            inputs['obs'] = torch.randn(B, obs_shape * 2)
        else:
            obs_shape = (obs_shape[0] * 2, *obs_shape[1:])
            inputs['obs'] = torch.randn(B, *obs_shape)
        # inputs['action'] = torch.randint(action_shape, B)
        if isinstance(action_shape, int):
            inputs['action'] = torch.randint(action_shape, (B, ))
        else:
            inputs['action'] = torch.randint(action_shape[0], (B, ))

        outputs = model.forward(inputs['obs'])
        assert isinstance(outputs, dict)
        if isinstance(action_shape, int):
            assert outputs['logit'].shape == (B, action_shape)
        else:
            assert outputs['logit'].shape == (B, *action_shape)
        # self.test_train(model, inputs)
        action = model.predict_action(inputs['obs'])['action']
        assert action.shape == (B, )

        loss = model.train(inputs, n_epoch=10, learning_rate=0.01, weight_decay=1e-4)
        assert isinstance(loss, float)

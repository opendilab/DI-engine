import pytest
from easydict import EasyDict
import torch
from ding.model.template import PDQN


@pytest.mark.unittest
class TestPQQN:

    def test_dqn(self):
        T, B = 3, 4
        obs_shape = (4, )
        act_shape = EasyDict({'action_type_shape': (3, ), 'action_args_shape': (5, )})
        if isinstance(obs_shape, int):
            cont_inputs = torch.randn(B, obs_shape)
        else:
            cont_inputs = torch.randn(B, *obs_shape)
        model = PDQN(obs_shape, act_shape)
        cont_outputs = model.forward(cont_inputs, mode='compute_continuous')
        assert isinstance(cont_outputs, dict)
        dis_inputs = {'state': cont_inputs, 'action_args': cont_outputs['action_args']}
        dis_outputs = model.forward(dis_inputs, mode='compute_discrete')
        assert isinstance(dis_outputs, dict)
        if isinstance(act_shape['action_type_shape'], int):
            assert dis_outputs['logit'].shape == (B, act_shape.action_type_shape)
        elif len(act_shape['action_type_shape']) == 1:
            assert dis_outputs['logit'].shape == (B, *act_shape.action_type_shape)
        else:
            for i, s in enumerate(act_shape):
                assert dis_outputs['logit'][i].shape == (B, s)

    def test_mdqn(self):
        T, B = 3, 4
        obs_shape = (4, )
        act_shape = EasyDict({'action_type_shape': 3, 'action_args_shape': 5})
        if isinstance(obs_shape, int):
            cont_inputs = torch.randn(B, obs_shape)
        else:
            cont_inputs = torch.randn(B, *obs_shape)
        model = PDQN(
            obs_shape, act_shape, multi_pass=True, action_mask=[[1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]]
        )
        cont_outputs = model.forward(cont_inputs, mode='compute_continuous')
        assert isinstance(cont_outputs, dict)
        dis_inputs = {'state': cont_inputs, 'action_args': cont_outputs['action_args']}

        dis_outputs = model.forward(dis_inputs, mode='compute_discrete')

        assert isinstance(dis_outputs, dict)
        if isinstance(act_shape['action_type_shape'], int):
            assert dis_outputs['logit'].shape == (B, act_shape.action_type_shape)
        elif len(act_shape['action_type_shape']) == 1:
            assert dis_outputs['logit'].shape == (B, *act_shape.action_type_shape)
        else:
            for i, s in enumerate(act_shape):
                assert dis_outputs['logit'][i].shape == (B, s)

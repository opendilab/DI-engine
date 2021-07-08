import copy
from copy import deepcopy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import logging

from ding.torch_utils import get_lstm
from ding.model import model_wrap, register_wrapper, IModelWrapper, BaseModelWrapper


class TempMLP(torch.nn.Module):

    def __init__(self):
        super(TempMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 6)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x


class ActorMLP(torch.nn.Module):

    def __init__(self):
        super(ActorMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.out = nn.Softmax()

    def forward(self, inputs, tmp=0):
        x = self.fc1(inputs['obs'])
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.out(x)
        ret = {'logit': x, 'tmp': tmp, 'action': x + torch.rand_like(x)}
        if 'mask' in inputs:
            ret['action_mask'] = inputs['mask']
        return ret


class TempLSTM(torch.nn.Module):

    def __init__(self):
        super(TempLSTM, self).__init__()
        self.model = get_lstm(lstm_type='pytorch', input_size=36, hidden_size=32, num_layers=2, norm_type=None)

    def forward(self, data):
        output, next_state = self.model(data['f'], data['prev_state'], list_next_state=True)
        return {'output': output, 'next_state': next_state}


@pytest.fixture(scope='function')
def setup_model():
    return torch.nn.Linear(3, 6)


@pytest.mark.unittest
class TestModelWrappers:

    def test_hidden_state_wrapper(self):

        model = TempLSTM()
        state_num = 4
        model = model_wrap(model, wrapper_name='hidden_state', state_num=state_num, save_prev_state=True)
        model.reset()
        data = {'f': torch.randn(2, 4, 36)}
        output = model.forward(data)
        assert output['output'].shape == (2, state_num, 32)
        assert output['prev_state'] == [None for _ in range(4)]
        for item in model._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 3, 36)}
        data_id = [0, 1, 3]
        output = model.forward(data, data_id=data_id)
        assert output['output'].shape == (2, 3, 32)
        assert all([len(s) == 2 for s in output['prev_state']])
        for item in model._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 2, 36)}
        data_id = [0, 1]
        output = model.forward(data, data_id=data_id)
        assert output['output'].shape == (2, 2, 32)

        assert all([isinstance(s, tuple) and len(s) == 2 for s in model._state.values()])
        model.reset()
        assert all([isinstance(s, type(None)) for s in model._state.values()])

    def test_target_network_wrapper(self):

        model = TempMLP()
        target_model = deepcopy(model)
        target_model2 = deepcopy(model)
        target_model = model_wrap(target_model, wrapper_name='target', update_type='assign', update_kwargs={'freq': 2})
        model = model_wrap(model, wrapper_name='base')
        register_wrapper('abstract', IModelWrapper)
        assert all([hasattr(target_model, n) for n in ['reset', 'forward', 'update']])
        assert model.fc1.weight.eq(target_model.fc1.weight).sum() == 12
        model.fc1.weight.data = torch.randn_like(model.fc1.weight)
        assert model.fc1.weight.ne(target_model.fc1.weight).sum() == 12
        target_model.update(model.state_dict(), direct=True)
        assert model.fc1.weight.eq(target_model.fc1.weight).sum() == 12
        model.reset()
        target_model.reset()

        inputs = torch.randn(2, 3)
        model.train()
        target_model.train()
        output = model.forward(inputs)
        with torch.no_grad():
            output_target = target_model.forward(inputs)
        assert output.eq(output_target).sum() == 2 * 6
        model.fc1.weight.data = torch.randn_like(model.fc1.weight)
        assert model.fc1.weight.ne(target_model.fc1.weight).sum() == 12
        target_model.update(model.state_dict())
        assert model.fc1.weight.ne(target_model.fc1.weight).sum() == 12
        target_model.update(model.state_dict())
        assert model.fc1.weight.eq(target_model.fc1.weight).sum() == 12

        target_model2 = model_wrap(
            target_model2, wrapper_name='target', update_type='momentum', update_kwargs={'theta': 0.01}
        )
        target_model2.update(model.state_dict(), direct=True)
        assert model.fc1.weight.eq(target_model2.fc1.weight).sum() == 12
        model.fc1.weight.data = torch.randn_like(model.fc1.weight)
        old_state_dict = target_model2.state_dict()
        target_model2.update(model.state_dict())
        assert target_model2.fc1.weight.data.eq(
            old_state_dict['fc1.weight'] * (1 - 0.01) + model.fc1.weight.data * 0.01
        ).all()

    def test_eps_greedy_wrapper(self):
        model = ActorMLP()
        model = model_wrap(model, wrapper_name='eps_greedy_sample')
        model.eval()
        eps_threshold = 0.5
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data, eps=eps_threshold)
        assert output['tmp'] == 0
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, eps=eps_threshold, tmp=1)
            assert isinstance(output, dict)
        assert output['tmp'] == 1

    def test_argmax_sample_wrapper(self):
        model = model_wrap(ActorMLP(), wrapper_name='argmax_sample')
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        logit = output['logit']
        assert output['action'].eq(logit.argmax(dim=-1)).all()
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = model.forward(data)
        logit = output['logit'].sub(1e8 * (1 - data['mask']))
        assert output['action'].eq(logit.argmax(dim=-1)).all()

    def test_multinomial_sample_wrapper(self):
        model = model_wrap(ActorMLP(), wrapper_name='multinomial_sample')
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        assert output['action'].shape == (4, )
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = model.forward(data)
        assert output['action'].shape == (4, )

    def test_action_noise_wrapper(self):
        model = model_wrap(
            ActorMLP(),
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_range={
                'min': -0.1,
                'max': 0.1
            },
            action_range={
                'min': -0.05,
                'max': 0.05
            }
        )
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        action = output['action']
        assert action.shape == (4, 6)
        assert action.eq(action.clamp(-0.05, 0.05)).all()

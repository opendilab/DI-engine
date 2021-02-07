import copy
from copy import deepcopy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from nervex.torch_utils import get_lstm
from nervex.armor import Armor, BaseArmor, IArmorStatelessPlugin, register_plugin


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
class TestBaseArmor:

    def test_naive(self, setup_model):
        armor = BaseArmor(setup_model)
        armor.mode(train=False)
        assert not armor.model.training
        armor.mode(train=True)
        assert armor.model.training
        armor.reset()
        state_dict = armor.state_dict()
        assert isinstance(state_dict, dict)
        armor.load_state_dict(state_dict)

        data = torch.randn(4, 3)
        output = armor.forward(data)
        assert output.shape == (4, 6)
        armor.model = None
        with pytest.raises(TypeError):
            armor.forward(data)


@pytest.mark.unittest
class TestArmorPlugin:

    def test_grad_helper(self, setup_model):
        armor1 = Armor(deepcopy(setup_model))
        armor1.add_plugin('main', 'grad', enable_grad=True)
        armor2 = Armor(deepcopy(setup_model))
        armor2.add_plugin('main', 'grad', enable_grad=False)

        data = torch.randn(4, 3).requires_grad_(True)
        assert armor1.model.weight.grad is None
        assert data.grad is None
        output = armor1.forward(data)
        loss = output.mean()
        loss.backward()
        assert isinstance(armor1.model.weight.grad, torch.Tensor)
        assert isinstance(data.grad, torch.Tensor)

        data = torch.randn(4, 3).requires_grad_(True)
        assert armor2.model.weight.grad is None
        assert data.grad is None
        output = armor2.forward(data)
        loss = output.mean()
        with pytest.raises(RuntimeError):
            loss.backward()
        loss += data.mean()
        loss.backward()
        assert armor2.model.weight.grad is None
        assert isinstance(data.grad, torch.Tensor)

    def test_hidden_state_helper(self):

        model = TempLSTM()
        state_num = 4
        # the former plugin is registered in inner layer
        armor = Armor(model)
        armor.add_plugin('main', 'hidden_state', state_num=state_num, save_prev_state=True)
        armor.add_plugin('main', 'grad', enable_grad=True)
        armor.reset()
        data = {'f': torch.randn(2, 4, 36)}
        output = armor.forward(data)
        assert output['output'].shape == (2, state_num, 32)
        assert output['prev_state'] == [None for _ in range(4)]
        for item in armor._state_manager._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 3, 36)}
        data_id = [0, 1, 3]
        output = armor.forward(data, data_id=data_id)
        assert output['output'].shape == (2, 3, 32)
        assert all([len(s) == 2 for s in output['prev_state']])
        for item in armor._state_manager._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 2, 36)}
        data_id = [0, 1]
        output = armor.forward(data, data_id=data_id)
        assert output['output'].shape == (2, 2, 32)

        assert all([isinstance(s, tuple) and len(s) == 2 for s in armor._state_manager._state.values()])
        armor.reset()
        assert all([isinstance(s, type(None)) for s in armor._state_manager._state.values()])

    def test_target_network_helper(self):

        model = TempMLP()
        armor = Armor(model)
        armor.add_model('target', update_type='assign', update_kwargs={'freq': 2})
        armor.add_plugin('main', 'grad', enable_grad=True)
        armor.add_plugin('target', 'grad', enable_grad=False)
        with pytest.raises(KeyError):
            armor.add_plugin('main', 'grad_error', enable_grad=False)
        register_plugin('abstract', IArmorStatelessPlugin)
        with pytest.raises(NotImplementedError):
            armor.add_plugin('main', 'abstract')
        assert all([hasattr(armor, n) for n in ['target_reset', 'target_mode', 'target_forward', 'target_update']])
        assert armor.model.fc1.weight.eq(armor.target_model.fc1.weight).sum() == 12
        armor.model.fc1.weight.data = torch.randn_like(armor.model.fc1.weight)
        assert armor.model.fc1.weight.ne(armor.target_model.fc1.weight).sum() == 12
        armor.target_update(armor.state_dict()['model'], direct=True)
        assert armor.model.fc1.weight.eq(armor.target_model.fc1.weight).sum() == 12
        armor.reset()
        armor.target_reset()

        inputs = torch.randn(2, 3)
        armor.mode(train=True)
        armor.target_mode(train=True)
        output = armor.forward(inputs)
        output_target = armor.target_forward(inputs)
        assert output.eq(output_target).sum() == 2 * 6
        armor.model.fc1.weight.data = torch.randn_like(armor.model.fc1.weight)
        assert armor.model.fc1.weight.ne(armor.target_model.fc1.weight).sum() == 12
        armor.target_update(armor.state_dict()['model'])
        assert armor.model.fc1.weight.ne(armor.target_model.fc1.weight).sum() == 12
        armor.target_update(armor.state_dict()['model'])
        assert armor.model.fc1.weight.eq(armor.target_model.fc1.weight).sum() == 12

        with pytest.raises(KeyError):
            armor.remove_model('target_error')
        armor.remove_model('target')
        with pytest.raises(AttributeError):
            armor.target_forward(inputs)
        with pytest.raises(NotImplementedError):
            armor.remove_plugin('main', 'naive')
        armor.add_model('target', update_type='momentum', update_kwargs={'theta': 0.01})
        assert armor.model.fc1.weight.eq(armor.target_model.fc1.weight).sum() == 12
        armor.model.fc1.weight.data = torch.randn_like(armor.model.fc1.weight)
        old_state_dict = armor.target_model.state_dict()
        armor.target_update(armor.state_dict()['model'])
        assert armor.target_model.fc1.weight.data.eq(
            old_state_dict['fc1.weight'] * (1 - 0.01) + armor.model.fc1.weight.data * 0.01
        ).all()

    def test_teacher_network_helper(self):
        model = TempLSTM()
        armor = Armor(model)
        state_num = 4
        armor.add_model('teacher', teacher_cfg={})
        armor.add_plugin('main', 'hidden_state', state_num=state_num, save_prev_state=True)
        armor.add_plugin('teacher', 'hidden_state', state_num=state_num, save_prev_state=True)
        assert all(
            [
                hasattr(armor, n)
                for n in ['teacher_reset', 'teacher_mode', 'teacher_forward', 'teacher_load_state_dict']
            ]
        )
        armor.mode(train=True)
        armor.teacher_mode(train=False)
        armor.reset()
        armor.teacher_reset()
        assert all([m.ne(t).sum() == 0 for m, t in zip(armor.model.parameters(), armor.teacher_model.parameters())])
        for p in armor.model.parameters():
            p.data = torch.randn_like(p)
        assert all([m.eq(t).sum() == 0 for m, t in zip(armor.model.parameters(), armor.teacher_model.parameters())])
        armor.teacher_load_state_dict({'model': armor.model.state_dict()})
        assert all([m.ne(t).sum() == 0 for m, t in zip(armor.model.parameters(), armor.teacher_model.parameters())])
        data = {'f': torch.randn(2, 4, 36)}
        output = armor.forward(data)
        assert set(output.keys()) == {'output', 'prev_state'}
        assert all([p is None for p in output['prev_state']])
        output = armor.teacher_forward(data)
        assert set(output.keys()) == {'output', 'prev_state'}
        assert all([p is None for p in output['prev_state']])
        data = {'f': torch.randn(2, 3, 36)}
        output_main = armor.forward(data, data_id=[0, 1, 2])
        output_teacher = armor.teacher_forward(data, data_id=[0, 1, 3])
        for i in [0, 1]:
            assert output_main['output'][:, i].ne(output_teacher['output'][:, i]).sum() == 0
        assert output_main['output'][:, 2].eq(output_teacher['output'][:, 2]).sum() == 0
        assert len(output_main['prev_state']) == 3
        assert all([len(p) == 2 for p in output_main['prev_state']])
        assert len(output_teacher['prev_state']) == 3
        assert all([len(p) == 2 for p in output_teacher['prev_state']])

    def test_eps_greedy_helper(self):
        model = ActorMLP()
        armor = Armor(model)
        armor.add_plugin('main', 'eps_greedy_sample')
        armor.add_plugin('main', 'grad', enable_grad=False)
        armor.mode(train=False)
        eps_threshold = 0.5
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = armor.forward(data, eps=eps_threshold)
        assert output['tmp'] == 0
        for i in range(10):
            if i == 5:
                data.pop('mask')
            output = armor.forward(data, eps=eps_threshold, param={'tmp': 1})
            assert isinstance(output, dict)
        assert output['tmp'] == 1

    def test_argmax_sample_helper(self):
        armor = Armor(ActorMLP())
        armor.add_plugin('main', 'argmax_sample')
        data = {'obs': torch.randn(4, 3)}
        output = armor.forward(data)
        logit = output['logit']
        assert output['action'].eq(logit.argmax(dim=-1)).all()
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = armor.forward(data)
        logit = output['logit'].sub(1e8 * (1 - data['mask']))
        assert output['action'].eq(logit.argmax(dim=-1)).all()

    def test_multinomial_sample_helper(self):
        armor = Armor(ActorMLP())
        armor.add_plugin('main', 'multinomial_sample')
        data = {'obs': torch.randn(4, 3)}
        output = armor.forward(data)
        assert output['action'].shape == (4, )
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = armor.forward(data)
        assert output['action'].shape == (4, )

    def test_action_noise_helper(self):
        armor = Armor(ActorMLP())
        armor.add_plugin(
            'main',
            'action_noise',
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
        output = armor.forward(data)
        action = output['action']
        assert action.shape == (4, 6)
        assert action.eq(action.clamp(-0.05, 0.05)).all()

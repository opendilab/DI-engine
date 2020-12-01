import copy
from copy import deepcopy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from nervex.torch_utils import get_lstm
from nervex.worker.agent.base_agent import BaseAgent, AgentAggregator


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
class TestBaseAgent:

    def test_naive(self, setup_model):
        agent = BaseAgent(setup_model, plugin_cfg=None)
        agent.mode(train=False)
        assert not agent.model.training
        agent.mode(train=True)
        assert agent.model.training
        state_dict = agent.state_dict()
        assert isinstance(state_dict, dict)
        agent.load_state_dict(state_dict)

        data = torch.randn(4, 3)
        output = agent.forward(data)
        assert output.shape == (4, 6)


@pytest.mark.unittest
class TestAgentPlugin:

    def test_grad_helper(self, setup_model):
        agent1 = BaseAgent(deepcopy(setup_model), plugin_cfg=OrderedDict({'grad': {'enable_grad': True}}))
        agent2 = BaseAgent(deepcopy(setup_model), plugin_cfg=OrderedDict({'grad': {'enable_grad': False}}))

        data = torch.randn(4, 3).requires_grad_(True)
        assert agent1.model.weight.grad is None
        assert data.grad is None
        output = agent1.forward(data)
        loss = output.mean()
        loss.backward()
        assert isinstance(agent1.model.weight.grad, torch.Tensor)
        assert isinstance(data.grad, torch.Tensor)

        data = torch.randn(4, 3).requires_grad_(True)
        assert agent2.model.weight.grad is None
        assert data.grad is None
        output = agent2.forward(data)
        loss = output.mean()
        with pytest.raises(RuntimeError):
            loss.backward()
        loss += data.mean()
        loss.backward()
        assert agent2.model.weight.grad is None
        assert isinstance(data.grad, torch.Tensor)

    def test_hidden_state_helper(self):

        model = TempLSTM()
        state_num = 4
        plugin_cfg = OrderedDict(
            {
                'hidden_state': {
                    'state_num': state_num,
                    'save_prev_state': True
                },
                'grad': {
                    'enable_grad': True
                }
            }
        )
        # the former plugin is registered in inner layer
        agent = BaseAgent(model, plugin_cfg)
        agent.reset()
        data = {'f': torch.randn(2, 4, 36)}
        output = agent.forward(data)
        assert output['output'].shape == (2, state_num, 32)
        assert output['prev_state'] == [None for _ in range(4)]
        for item in agent._state_manager._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 3, 36)}
        state_id = [0, 1, 3]
        output = agent.forward(data, state_id=state_id)
        assert output['output'].shape == (2, 3, 32)
        assert all([len(s) == 2 for s in output['prev_state']])
        for item in agent._state_manager._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 2, 36)}
        state_id = [0, 1]
        output = agent.forward(data, state_id=state_id)
        assert output['output'].shape == (2, 2, 32)

        assert all([isinstance(s, tuple) and len(s) == 2 for s in agent._state_manager._state.values()])
        agent.reset()
        assert all([isinstance(s, type(None)) for s in agent._state_manager._state.values()])

    def test_target_network_helper(self):

        model = TempMLP()
        plugin_cfg = {
            'main': OrderedDict({'grad': {
                'enable_grad': True
            }}),
            'target': OrderedDict(
                {
                    'target': {
                        'update_type': 'assign',
                        'kwargs': {
                            'freq': 10
                        }
                    },
                    'grad': {
                        'enable_grad': False
                    }
                }
            )
        }
        agent = AgentAggregator(BaseAgent, model, plugin_cfg)
        assert all([hasattr(agent, n) for n in ['target_reset', 'target_mode', 'target_forward', 'target_update']])
        assert agent.model.fc1.weight.eq(agent.target_model.fc1.weight).sum() == 12
        agent.model.fc1.weight.data = torch.randn_like(agent.model.fc1.weight)
        assert agent.model.fc1.weight.ne(agent.target_model.fc1.weight).sum() == 12
        agent.target_update(agent.state_dict()['model'], direct=True)
        assert agent.model.fc1.weight.eq(agent.target_model.fc1.weight).sum() == 12

        inputs = torch.randn(2, 3)
        agent.mode(train=True)
        agent.target_mode(train=True)
        output = agent.forward(inputs)
        output_target = agent.target_forward(inputs)
        assert output.eq(output_target).sum() == 2 * 6

    @pytest.mark.u
    def test_teacher_network_helper(self):
        model = TempLSTM()
        plugin_cfg = {
            'main': OrderedDict({'hidden_state': {
                'state_num': 4,
                'save_prev_state': True
            }}),
            'teacher': OrderedDict(
                {
                    'hidden_state': {
                        'state_num': 4,
                        'save_prev_state': True
                    },
                    'teacher': {
                        'teacher_cfg': {}
                    }
                }
            )
        }
        agent = AgentAggregator(BaseAgent, model, plugin_cfg)
        assert all(
            [
                hasattr(agent, n)
                for n in ['teacher_reset', 'teacher_mode', 'teacher_forward', 'teacher_load_state_dict']
            ]
        )
        agent.mode(train=True)
        agent.teacher_mode(train=False)
        agent.reset()
        agent.teacher_reset()
        assert all([m.ne(t).sum() == 0 for m, t in zip(agent.model.parameters(), agent.teacher_model.parameters())])
        for p in agent.model.parameters():
            p.data = torch.randn_like(p)
        assert all([m.eq(t).sum() == 0 for m, t in zip(agent.model.parameters(), agent.teacher_model.parameters())])
        agent.teacher_load_state_dict({'model': agent.model.state_dict()})
        assert all([m.ne(t).sum() == 0 for m, t in zip(agent.model.parameters(), agent.teacher_model.parameters())])
        data = {'f': torch.randn(2, 4, 36)}
        output = agent.forward(data)
        assert set(output.keys()) == {'output', 'prev_state'}
        assert all([p is None for p in output['prev_state']])
        output = agent.teacher_forward(data)
        assert set(output.keys()) == {'output', 'prev_state'}
        assert all([p is None for p in output['prev_state']])
        data = {'f': torch.randn(2, 3, 36)}
        output_main = agent.forward(data, state_id=[0, 1, 2])
        output_teacher = agent.teacher_forward(data, state_id=[0, 1, 3])
        for i in [0, 1]:
            assert output_main['output'][:, i].ne(output_teacher['output'][:, i]).sum() == 0
        assert output_main['output'][:, 2].eq(output_teacher['output'][:, 2]).sum() == 0
        assert len(output_main['prev_state']) == 3
        assert all([len(p) == 2 for p in output_main['prev_state']])
        assert len(output_teacher['prev_state']) == 3
        assert all([len(p) == 2 for p in output_teacher['prev_state']])

import copy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from nervex.torch_utils import get_lstm
from nervex.worker.agent.base_agent import BaseAgent


@pytest.fixture(scope='function')
def setup_model():
    return torch.nn.Linear(3, 6)


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


class TestAgentPlugin:

    def test_grad_helper(self, setup_model):
        agent1 = BaseAgent(copy.deepcopy(setup_model), plugin_cfg=OrderedDict({'grad': {'enable_grad': True}}))
        agent2 = BaseAgent(copy.deepcopy(setup_model), plugin_cfg=OrderedDict({'grad': {'enable_grad': False}}))

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

        class TempLSTM(torch.nn.Module):

            def __init__(self):
                super(TempLSTM, self).__init__()
                self.model = get_lstm(lstm_type='pytorch', input_size=36, hidden_size=32, num_layers=2, norm_type=None)

            def forward(self, data):
                return self.model(data['f'], data['prev_state'], list_next_state=True)

        model = TempLSTM()
        state_num = 4
        plugin_cfg = OrderedDict({
            'hidden_state': {
                'state_num': state_num
            },
            'grad': {
                'enable_grad': True
            },
        })
        # the former plugin is registered in inner layer
        agent = BaseAgent(model, plugin_cfg)
        data = {'f': torch.randn(2, 4, 36)}
        state_info = {i: False for i in range(state_num)}
        output = agent.forward(data, state_info=state_info)
        assert output.shape == (2, state_num, 32)
        for item in agent._state_manager._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 3, 36)}
        state_info = {i: False for i in [0, 2, 3]}
        output = agent.forward(data, state_info=state_info)
        assert output.shape == (2, 3, 32)
        for item in agent._state_manager._state.values():
            assert isinstance(item, tuple) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item)

        data = {'f': torch.randn(2, 2, 36)}
        state_info = {i: False for i in [0, 1]}
        output = agent.forward(data, state_info=state_info)
        assert output.shape == (2, 2, 32)

        assert all([isinstance(s, tuple) and len(s) == 2 for s in agent._state_manager._state.values()])
        agent.reset()
        assert all([isinstance(s, type(None)) for s in agent._state_manager._state.values()])

    def test_target_network_helper(self):

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

        model = TempMLP()
        plugin_cfg = {
            'grad': {
                'enable_grad': True
            },
            'target_network': {
                'update_cfg': {
                    'type': 'assign',
                    'kwargs': {
                        'freq': 10
                    },
                }
            }
        }
        plugin_cfg = OrderedDict(plugin_cfg)
        agent = BaseAgent(model, plugin_cfg)
        assert all(
            [hasattr(agent, n) for n in ['_target_network', 'target_mode', 'target_forward', 'update_target_network']]
        )
        assert agent.model.fc1.weight.eq(agent._target_network._model.fc1.weight).sum() == 12
        agent.model.fc1.weight.data = torch.randn_like(agent.model.fc1.weight)
        assert agent.model.fc1.weight.ne(agent._target_network._model.fc1.weight).sum() == 12
        agent.update_target_network(agent.state_dict()['model'], direct=True)

        inputs = torch.randn(2, 3)
        agent.mode(train=True)
        agent.target_mode(train=True)
        output = agent.forward(inputs)
        output_target = agent.target_forward(inputs)
        assert output.eq(output_target).sum() == 2 * 6

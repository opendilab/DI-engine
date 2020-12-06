import copy
from copy import deepcopy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from nervex.model import ConvValueAC
from nervex.worker.agent.base_agent import BaseAgent, AgentAggregator
from nervex.worker.agent.agent_template import create_ac_actor_agent, create_ac_evaluator_agent, \
    create_ac_learner_agent, \
    create_dqn_actor_agent, create_dqn_evaluator_agent, create_dqn_learner_agent, create_drqn_actor_agent, \
    create_drqn_evaluator_agent, create_drqn_learner_agent


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

    def forward(self, x, mode=None):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.out(x)
        if mode is not None:
            return {'logit': x, "action": 1}
        return {'logit': x}


@pytest.mark.unittest
class TestCreates:

    def test_create_ac_agent(self):
        model = ActorMLP()
        ac_actor_agent = create_ac_actor_agent(model)
        assert issubclass(type(ac_actor_agent), AgentAggregator)
        data = torch.randn(4, 3)
        output = ac_actor_agent.forward(data)
        assert output
        ac_evaluator_agent = create_ac_evaluator_agent(model)
        output = ac_evaluator_agent.forward(data)
        assert output
        assert issubclass(type(ac_evaluator_agent), AgentAggregator)
        ac_learner_agent = create_ac_learner_agent(model)
        assert issubclass(type(ac_learner_agent), AgentAggregator)

    def test_create_dqn_agent(self):
        model = TempMLP()
        dqn_actor_agent = create_dqn_actor_agent(model)
        assert issubclass(type(dqn_actor_agent), AgentAggregator)
        dqn_evaluator_agent = create_dqn_evaluator_agent(model)
        assert issubclass(type(dqn_evaluator_agent), AgentAggregator)
        dqn_learner_agent = create_dqn_learner_agent(model)
        assert issubclass(type(dqn_learner_agent), AgentAggregator)

    def test_create_drqn_agent(self):
        model = TempMLP()
        drqn_actor_agent = create_drqn_actor_agent(model, 3)
        assert issubclass(type(drqn_actor_agent), AgentAggregator)
        drqn_evaluator_agent = create_drqn_evaluator_agent(model, 3)
        assert issubclass(type(drqn_evaluator_agent), AgentAggregator)
        drqn_learner_agent = create_drqn_learner_agent(model, 3)
        assert issubclass(type(drqn_learner_agent), AgentAggregator)

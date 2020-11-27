import torch
import copy
from typing import Any, Optional
from collections import OrderedDict
from nervex.worker.agent import BaseAgent, AgentAggregator


# ######################## Learner ######################################
def create_dqn_learner_agent(model: torch.nn.Module, is_double: bool = True) -> BaseAgent:
    plugin_cfg = OrderedDict({
        'grad': {
            'enable_grad': True
        },
    })
    # whether use double(target) q-network plugin
    if is_double:
        # plugin_cfg['target'] = {'update_cfg': {'type': 'momentum', 'kwargs': {'theta': 0.001}}}
        plugin_cfg['target'] = {'update_cfg': {'type': 'assign', 'kwargs': {'freq': 500}}}
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    agent.is_double = is_double
    return agent


def create_drqn_learner_agent(model: torch.nn.Module, state_num: int, is_double: bool = True) -> BaseAgent:
    plugin_cfg = OrderedDict({
        'hidden_state': {
            'state_num': state_num,
        },
        'grad': {
            'enable_grad': True
        },
    })
    # whether use double(target) q-network plugin
    if is_double:
        plugin_cfg['target'] = {'update_cfg': {'type': 'momentum', 'kwargs': {'theta': 0.001}}}
        # plugin_cfg['target'] = {'update_cfg': {'type': 'assign', 'kwargs': {'freq': 500}}}
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    agent.is_double = is_double
    return agent


class ACAgent(BaseAgent):
    """
    Overview:
        Actor-Critic agent (for both learner and actor)
    """

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'compute_action_value'
        return super().forward(data, param)


def create_ac_learner_agent(model: torch.nn.Module) -> ACAgent:
    plugin_cfg = OrderedDict({
        'grad': {
            'enable_grad': True
        },
    })
    agent = AgentAggregator(ACAgent, model, plugin_cfg)
    return agent


def create_qac_learner_agent(model: torch.nn.Module, is_double: bool = True) -> BaseAgent:
    plugin_cfg = OrderedDict({
        'action_noise': {},
        'grad': {
            'enable_grad': True
        },
    })
    if is_double:
        plugin_cfg['target'] = {'update_cfg': {'type': 'assign', 'kwargs': {'freq': 500}}}
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    agent.is_double = is_double
    return agent


# ######################## Actor ######################################


def create_dqn_actor_agent(model: torch.nn.Module) -> BaseAgent:
    plugin_cfg = OrderedDict({
        'eps_greedy_sample': {},
        'grad': {
            'enable_grad': False
        },
    })
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_drqn_actor_agent(model: torch.nn.Module, state_num: int) -> BaseAgent:
    plugin_cfg = OrderedDict(
        {
            'hidden_state': {
                'state_num': state_num,
                'save_prev_state': True,
            },
            'eps_greedy_sample': {},
            'grad': {
                'enable_grad': False
            },
        }
    )
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_ac_actor_agent(model: torch.nn.Module) -> ACAgent:
    plugin_cfg = OrderedDict({
        'multinomial_sample': {},
        'grad': {
            'enable_grad': False,
        },
    })
    agent = AgentAggregator(ACAgent, model, plugin_cfg)
    return agent


def create_qac_actor_agent(model: torch.nn.Module) -> BaseAgent:
    plugin_cfg = OrderedDict({
        'action_noise': {},
        'grad': {
            'enable_grad': False
        },
    })
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


# ######################## Evaluator ######################################


def create_dqn_evaluator_agent(model: torch.nn.Module) -> BaseAgent:
    plugin_cfg = OrderedDict({
        'argmax_sample': {},
        'grad': {
            'enable_grad': False
        },
    })
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_drqn_evaluator_agent(model: torch.nn.Module, state_num: int) -> BaseAgent:
    plugin_cfg = OrderedDict(
        {
            'hidden_state': {
                'state_num': state_num,
            },
            'argmax_sample': {},
            'grad': {
                'enable_grad': False
            },
        }
    )
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


class ACEvaluatorAgent(BaseAgent):

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'compute_action'
        return super().forward(data, param)


def create_ac_evaluator_agent(model: torch.nn.Module) -> ACEvaluatorAgent:
    plugin_cfg = OrderedDict({
        'argmax_sample': {},
        'grad': {
            'enable_grad': False,
        },
    })
    agent = AgentAggregator(ACEvaluatorAgent, model, plugin_cfg)
    return agent


def create_qac_evaluator_agent(model: torch.nn.Module) -> BaseAgent:
    plugin_cfg = OrderedDict({
        'grad': {
            'enable_grad': False,
        },
    })
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent

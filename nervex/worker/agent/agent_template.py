import torch
import copy
from typing import Any, Optional
from collections import OrderedDict
from nervex.worker.agent import BaseAgent, AgentAggregator


# ######################## Learner ######################################
def create_dqn_learner_agent(model: torch.nn.Module, is_double: bool = True) -> BaseAgent:
    r"""
    Overview:
        create the dqn learner agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent
        - is_double (:obj:`bool`): whether the learner agent will use double network

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the dqn_actor_agent
    """
    plugin_cfg = {'main': OrderedDict({'grad': {'enable_grad': True}})}
    # whether use double(target) q-network plugin
    if is_double:
        # plugin_cfg['target'] = OrderedDict({'target': {'update_type': 'momentum', 'kwargs': {'theta': 0.001}},
        #   'grad': {'enable_grad': False}})
        plugin_cfg['target'] = OrderedDict(
            {
                'target': {
                    'update_type': 'assign',
                    'kwargs': {
                        'freq': 500
                    }
                },
                'grad': {
                    'enable_grad': False
                }
            }
        )
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    agent.is_double = is_double
    return agent


def create_drqn_learner_agent(model: torch.nn.Module, state_num: int, is_double: bool = True) -> BaseAgent:
    r"""
    Overview:
        create the drqn learner agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent
        - state_num (:obj:`int`): the num of hidden states in the drqn_learner_agent
        - is_double (:obj:`bool`): whether the learner agent will use double network

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the drqn_actor_agent
    """
    plugin_cfg = {'main': OrderedDict({'hidden_state': {'state_num': state_num}, 'grad': {'enable_grad': True}})}
    # whether use double(target) q-network plugin
    if is_double:
        plugin_cfg['target'] = OrderedDict(
            {
                'hidden_state': {
                    'state_num': state_num
                },
                'target': {
                    'update_type': 'momentum',
                    'kwargs': {
                        'theta': 0.001
                    }
                },
                'grad': {
                    'enable_grad': False
                }
            }
        )
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    agent.is_double = is_double
    return agent


def create_qmix_learner_agent(model: torch.nn.Module, state_num: int, agent_num: int) -> BaseAgent:
    plugin_cfg = {'main': OrderedDict({'hidden_state': {'state_num': state_num}, 'grad': {'enable_grad': True}})}
    plugin_cfg['target'] = OrderedDict(
        {
            'hidden_state': {
                'state_num': state_num,
                'init_fn': lambda: [None for _ in range(agent_num)],
            },
            'target': {
                'update_type': 'momentum',
                'kwargs': {
                    'theta': 0.001
                }
            },
            'grad': {
                'enable_grad': False
            }
        }
    )
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
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
    r"""
    Overview:
        create the actor critic learner agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the ac_learner_agent
    """
    plugin_cfg = {'main': OrderedDict({'grad': {'enable_grad': True}})}
    agent = AgentAggregator(ACAgent, model, plugin_cfg)
    return agent


def create_qac_learner_agent(
        model: torch.nn.Module, use_noise: bool = False, action_range: Optional[dict] = None
) -> BaseAgent:
    plugin_cfg = {'main': OrderedDict({'grad': {'enable_grad': True}})}
    plugin_cfg['target'] = OrderedDict(
        {
            'target': {
                'update_type': 'momentum',
                'kwargs': {
                    'theta': 0.005
                }
            },
            'grad': {
                'enable_grad': False
            }
        }
    )
    if use_noise:
        plugin_cfg['target']['action_noise'] = {
            'noise_type': 'gauss',
            'noise_kwargs': {
                'mu': 0.0,
                'sigma': 0.1,
            },
            'noise_range': {
                'min': -0.5,
                'max': 0.5
            },
            'action_range': action_range
        }
        plugin_cfg['target'].move_to_end('action_noise', last=False)  # move to beginning
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


# ######################## Actor ######################################


def create_dqn_actor_agent(model: torch.nn.Module) -> BaseAgent:
    r"""
    Overview:
        create the dqn actor agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the dqn_actor_agent
    """
    plugin_cfg = {'main': OrderedDict({'eps_greedy_sample': {}, 'grad': {'enable_grad': False}})}
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_drqn_actor_agent(model: torch.nn.Module, state_num: int) -> BaseAgent:
    r"""
    Overview:
        create the drqn actor agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the drqn_actor_agent
    """
    plugin_cfg = {
        'main': OrderedDict(
            {
                'hidden_state': {
                    'state_num': state_num,
                    'save_prev_state': True
                },
                'eps_greedy_sample': {},
                'grad': {
                    'enable_grad': False
                }
            }
        )
    }
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_qmix_actor_agent(model: torch.nn.Module, state_num: int, agent_num: int) -> BaseAgent:
    plugin_cfg = {
        'main': OrderedDict(
            {
                'hidden_state': {
                    'state_num': state_num,
                    'save_prev_state': True,
                    'init_fn': lambda: [None for _ in range(agent_num)],
                },
                'eps_greedy_sample': {},
                'grad': {
                    'enable_grad': False
                }
            }
        )
    }
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_ac_actor_agent(model: torch.nn.Module) -> ACAgent:
    r"""
    Overview:
        create the actor critic actor agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the ac_actor_agent
    """
    plugin_cfg = {'main': OrderedDict({'multinomial_sample': {}, 'grad': {'enable_grad': False}})}
    agent = AgentAggregator(ACAgent, model, plugin_cfg)
    return agent


def create_qac_actor_agent(model: torch.nn.Module, action_range: Optional[dict] = None) -> BaseAgent:
    plugin_cfg = {
        'main': OrderedDict(
            {
                'action_noise': {
                    'noise_type': 'gauss',
                    'noise_kwargs': {
                        'mu': 0.0,
                        'sigma': 0.2,
                    },
                    'noise_range': None,
                    'action_range': action_range
                },
                'grad': {
                    'enable_grad': False
                },
            }
        )
    }
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


# ######################## Evaluator ######################################


def create_dqn_evaluator_agent(model: torch.nn.Module) -> BaseAgent:
    r"""
    Overview:
        create the dqn evaluor agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the dqn_evaluator_agent
    """
    plugin_cfg = {'main': OrderedDict({'argmax_sample': {}, 'grad': {'enable_grad': False}})}
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_drqn_evaluator_agent(model: torch.nn.Module, state_num: int) -> BaseAgent:
    r"""
    Overview:
        create the drqn evaluor agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the drqn_evaluator_agent
    """
    plugin_cfg = {
        'main': OrderedDict(
            {
                'hidden_state': {
                    'state_num': state_num
                },
                'argmax_sample': {},
                'grad': {
                    'enable_grad': False
                }
            }
        )
    }
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


def create_qmix_evaluator_agent(model: torch.nn.Module, state_num: int, agent_num: int) -> BaseAgent:
    plugin_cfg = {
        'main': OrderedDict(
            {
                'hidden_state': {
                    'state_num': state_num,
                    'init_fn': lambda: [None for _ in range(agent_num)],
                },
                'argmax_sample': {},
                'grad': {
                    'enable_grad': False
                }
            }
        )
    }
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent


class ACEvaluatorAgent(BaseAgent):

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'compute_action'
        return super().forward(data, param)


def create_ac_evaluator_agent(model: torch.nn.Module) -> ACEvaluatorAgent:
    r"""
    Overview:
        create the actor critic evaluor agent

    Arguments:
        - model (:obj:`torch.nn.Module`): the model used in agent

    Returns:
        - agent (subclass of :obj:`BaseAgent`): the ac_evaluator_agent
    """
    plugin_cfg = {'main': OrderedDict({'argmax_sample': {}, 'grad': {'enable_grad': False}})}
    agent = AgentAggregator(ACEvaluatorAgent, model, plugin_cfg)
    return agent


def create_qac_evaluator_agent(model: torch.nn.Module) -> BaseAgent:
    plugin_cfg = {
        'main': OrderedDict({
            'grad': {
                'enable_grad': False,
            },
        })
    }
    agent = AgentAggregator(BaseAgent, model, plugin_cfg)
    return agent

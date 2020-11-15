import torch
from typing import Any, Optional
from collections import OrderedDict
from nervex.worker.agent import BaseAgent


# ######################## Learner ######################################
class DqnLearnerAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module, is_double: bool = True) -> None:
        self.plugin_cfg = OrderedDict({
            'grad': {
                'enable_grad': True
            },
        })
        # whether use double(target) q-network plugin
        if is_double:
            # self.plugin_cfg['target_network'] = {'update_cfg': {'type': 'momentum', 'kwargs': {'theta': 0.001}}}
            self.plugin_cfg['target_network'] = {'update_cfg': {'type': 'assign', 'kwargs': {'freq': 500}}}
        self.is_double = is_double
        super(DqnLearnerAgent, self).__init__(model, self.plugin_cfg)


class ACLearnerAgent(BaseAgent):
    """
    Overview:
        Actor-Critic learner agent
    """

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'grad': {
                'enable_grad': True
            },
        })
        super(ACLearnerAgent, self).__init__(model, plugin_cfg)

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'compute_action_value'
        return super().forward(data, param)


# ######################## Actor ######################################


class DqnActorAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'eps_greedy_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(DqnActorAgent, self).__init__(model, plugin_cfg)


class ACActorAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'multinomial_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(ACActorAgent, self).__init__(model, plugin_cfg)

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'compute_action_value'
        return super().forward(data, param)


# ######################## Evaluator ######################################


class DiscreteEvaluatorAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'argmax_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(DiscreteEvaluatorAgent, self).__init__(model, plugin_cfg)


class ACDiscreteEvaluatorAgent(DiscreteEvaluatorAgent):

    def forward(self, data: Any, param: Optional[dict] = None) -> dict:
        if param is None:
            param = {}
        param['mode'] = 'compute_action'
        return super().forward(data, param)

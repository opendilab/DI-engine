from collections import OrderedDict

import torch

from nervex.worker.agent import BaseAgent, IAgentStatelessPlugin, add_plugin


class SumoDqnLearnerAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module, plugin_cfg: dict) -> None:
        self.plugin_cfg = OrderedDict({
            'grad': {
                'enable_grad': True
            },
        })
        self.is_double = plugin_cfg['is_double']
        if plugin_cfg['is_double']:
            self.plugin_cfg['target_network'] = {'update_cfg': {'type': 'momentum', 'kwargs': {'theta': 0.001}}}
            # self.plugin_cfg['target_network'] = {'update_cfg': {'type': 'assign', 'kwargs': {'freq': 100}}}
        super(SumoDqnLearnerAgent, self).__init__(model, self.plugin_cfg)


class SumoDqnActorAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'eps_greedy_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(SumoDqnActorAgent, self).__init__(model, plugin_cfg)


class SumoDqnEvaluateAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'argmax_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(SumoDqnEvaluateAgent, self).__init__(model, plugin_cfg)

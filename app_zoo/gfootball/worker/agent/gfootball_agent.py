from collections import OrderedDict

import torch

from nervex.worker.agent import BaseAgent, IAgentStatelessPlugin, add_plugin


class GfootballIqlLearnerAgent(BaseAgent):

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
        super(GfootballIqlLearnerAgent, self).__init__(model, self.plugin_cfg)


class GfootballIqlActorAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict(
            {
                'eps_greedy_sample': {},
                'grad': {
                    'enable_grad': False
                },
            }
        )
        super(GfootballIqlActorAgent, self).__init__(model, plugin_cfg)


class GfootballIqlEvaluateAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict(
            {
                'argmax_sample': {},
                'grad': {
                    'enable_grad': False
                },
            }
        )
        super(GfootballIqlEvaluateAgent, self).__init__(model, plugin_cfg)

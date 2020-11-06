import torch
from collections import OrderedDict
from nervex.worker.agent import BaseAgent


class AtariDqnLearnerAgent(BaseAgent):

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
        super(AtariDqnLearnerAgent, self).__init__(model, self.plugin_cfg)


class AtariActorAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'eps_greedy_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(AtariActorAgent, self).__init__(model, plugin_cfg)


class AtariEvaluateAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'argmax_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(AtariEvaluateAgent, self).__init__(model, plugin_cfg)

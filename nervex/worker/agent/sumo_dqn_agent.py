import torch
import torch.nn as nn

from nervex.worker.agent import BaseAgent
from collections import OrderedDict


class SumoDqnAgent(BaseAgent):
    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'grad': {
                'enable_grad': True
            },
        })
        super(SumoDqnAgent, self).__init__(model, plugin_cfg)


class SumoDqnActorAgent(BaseAgent):
    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'grad': {
                'enable_grad': False
            },
        })
        super(SumoDqnActorAgent, self).__init__(model, plugin_cfg)

    def forward(self, *args, **kwargs):
        data = args[0]
        data = torch.stack(data, dim=0)
        return super().forward(data, **kwargs)

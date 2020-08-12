import torch
from collections import OrderedDict
from sc2learner.worker.agent import BaseAgent


class AlphaStarAgent(BaseAgent):
    def __init__(self, model: torch.nn.Module, env_num: int) -> None:
        plugin_cfg = OrderedDict({
            'hidden_state': {
                'state_num': env_num
            },
            'grad': {
                'enable_grad': False
            },
        })
        super(AlphaStarAgent, self).__init__(model, plugin_cfg)

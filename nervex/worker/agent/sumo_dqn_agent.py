import torch
import torch.nn as nn

from nervex.worker.agent import BaseAgent, IAgentStatelessPlugin, add_plugin
from collections import OrderedDict


class DataTransformHelper(IAgentStatelessPlugin):
    @classmethod
    def register(cls: type, agent: BaseAgent, ret_num: int):
        def data_wrapper(fn):
            def wrapper(*args, **kwargs):
                data = args[0]
                data = torch.stack(data, dim=0)
                ret = fn(data, **kwargs)
                # tl_num, bs -> bs, tl_num
                if ret_num == 1:
                    return list(zip(*ret))
                else:
                    return [list(zip(*r)) for r in ret]

            return wrapper

        agent.forward = data_wrapper(agent.forward)


add_plugin('sumowj3_data_transform', DataTransformHelper)


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
        plugin_cfg = OrderedDict(
            {
                'eps_greedy_sample': {},
                'sumowj3_data_transform': {
                    'ret_num': 2
                },
                'grad': {
                    'enable_grad': False
                },
            }
        )
        super(SumoDqnActorAgent, self).__init__(model, plugin_cfg)

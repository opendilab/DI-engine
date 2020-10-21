from collections import OrderedDict

import torch

from nervex.worker.agent import BaseAgent, IAgentStatelessPlugin, add_plugin


class DataTransformHelper(IAgentStatelessPlugin):

    @classmethod
    def register(cls: type, agent: BaseAgent):

        def data_wrapper(fn):

            def wrapper(*args, **kwargs):
                data = args[0]
                ret = fn(data, **kwargs)
                # tl_num, bs -> bs, tl_num
                result = list()
                for r in ret:
                    if isinstance(r, torch.Tensor) and len(r.shape) == 1:
                        r = [r]
                    result.append(list(zip(*r)))
                return result

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
                'sumowj3_data_transform': {},
                'grad': {
                    'enable_grad': False
                },
            }
        )
        super(SumoDqnActorAgent, self).__init__(model, plugin_cfg)


class SumoDqnEvaluateAgent(BaseAgent):

    def __init__(self, model: torch.nn.Module) -> None:
        plugin_cfg = OrderedDict({
            'argmax_sample': {},
            'sumowj3_data_transform': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(SumoDqnEvaluateAgent, self).__init__(model, plugin_cfg)

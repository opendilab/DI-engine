import torch
import copy
from torch.utils.data._utils.collate import default_collate
from collections import OrderedDict
from sc2learner.utils import list_dict2dict_list
from sc2learner.worker.agent import BaseAgent, add_plugin, IAgentStatelessPlugin


def as_eval_collate_fn(data):
    assert isinstance(data, list) or isinstance(data, tuple), type(data)
    data_item = {
        'spatial_info': True,
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'prev_state': False,
        'map_size': False,
    }
    new_data = list_dict2dict_list(data)
    # for the keys which are not in data_item, they'll be discarded
    for k, merge in data_item.items():
        if merge:
            new_data[k] = default_collate(new_data[k])
    return new_data


def post_processing(data, prev_state, bs):
    action, action_output, h = data
    action_output = [{k: action_output[k][b] for k in action_output.keys()} for b in range(bs)]
    entity_raw = [action['entity_raw'][b] for b in range(bs)]
    env_action = [{k: action['action'][k][b] for k in action['action'].keys()} for b in range(bs)]
    algo_action = copy.deepcopy(env_action)

    output = {}
    output['prev_state'] = prev_state
    output['action_output'] = action_output
    output['env_action'] = [{'action': a, 'entity_raw': e} for a, e in zip(env_action, entity_raw)]
    output['algo_action'] = algo_action
    return output, h


class ASDataTransformPlugin(IAgentStatelessPlugin):
    @classmethod
    def register(cls: type, agent: BaseAgent) -> None:
        def data_transform_wrapper(fn):
            def wrapper(data, **kwargs):
                prev_state = [d['prev_state'] for d in data]
                data = as_eval_collate_fn(data)
                ret = fn(data, **kwargs)
                return post_processing(ret, prev_state, bs=len(ret[2]))
            return wrapper

        agent.forward = data_transform_wrapper(agent.forward)


add_plugin('as_data_transform', ASDataTransformPlugin)


class AlphaStarAgent(BaseAgent):
    def __init__(self, model: torch.nn.Module, env_num: int) -> None:
        plugin_cfg = OrderedDict({
            'as_data_transform': {},
            'hidden_state': {
                'state_num': env_num
            },
            'grad': {
                'enable_grad': False
            },
        })
        super(AlphaStarAgent, self).__init__(model, plugin_cfg)

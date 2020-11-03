from collections import namedtuple

import torch

from nervex.envs.common import EnvElement


class GfootballRawAction(EnvElement):
    '''
    For raw action set please reference
    <https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#default-action-set>.
    '''
    _name = "gfootballRawAction"
    _action_keys = ['action_type']
    Action = namedtuple('Action', _action_keys)

    def _init(self, cfg):
        self._default_val = None
        self.template = {
            'action_type': {
                'name': 'action_type',
                'shape': (19, ),
                'value': {
                    'min': 0,
                    'max': 18,
                    'dtype': torch.LongTensor,
                    'dinfo': 'int value',
                },
                'env_value': 'type of action, refer to AtariEnv._action_set',
                'to_agent_processor': lambda x: x,
                'from_agent_processor': lambda x: x,
                'necessary': True,
            }
        }
        self._shape = (19, )
        self._value = {
            'min': 0,
            'max': 18,
            'dtype': torch.LongTensor,
            'dinfo': 'int value, action_meanings: []',
        }

    def _to_agent_processor(self, action):
        return action

    def _from_agent_processor(self, action):
        return action

    # override
    def _details(self):
        return '\t'.join(self._action_keys)

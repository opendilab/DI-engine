import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class PongRawAction(EnvElement):
    _name = "pongRawAction"
    _action_keys = ['action_type']
    Action = namedtuple('Action', _action_keys)

    action_meanings = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

    def _init(self):
        self._default_val = None
        self.template = {
            'action_type': {
                'name': 'action_type',
                'shape': (1, ),
                'value': {
                    'min': 0,
                    'max': 5,
                    'dtype': int,
                    'dinfo': 'int value',
                },
                'env_value': 'type of action, refer to AtariEnv._action_set',
                'to_agent_processor': lambda x: x,
                'from_agent_processor': lambda x: x,
                'necessary': True,
            }
        }
        self._shape = (1, )
        self._value = {
            'min': 0,
            'max': 5,
            'dtype': int,
            'dinfo': 'int value, action_meanings:{NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE}',
        }

    def _to_agent_processor(self, action):
        return action

    def _from_agent_processor(self, action):
        return action

    # override
    def _details(self):
        return '\t'.join(self._action_keys)

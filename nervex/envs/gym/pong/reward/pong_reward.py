import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class PongReward(EnvElement):
    _name = "pongReward"
    _reward_keys = ['reward_value']
    Reward = namedtuple('Action', _reward_keys)

    MinReward = -1.0
    MaxReward = 1.0

    def _init(self) -> None:
        self._default_val = 0.0
        self.template = {
            'reward_value': {
                'name': 'reward_value',
                'shape': (1, ),
                'value': {
                    'min': -1.0,
                    'max': 1.0,
                    'dtype': float,
                    'dinfo': 'float value',
                },
                'env_value': 'reward of action',
                'to_agent_processor': lambda x: x,
                'from_agent_processor': lambda x: x,
                'necessary': True,
            }
        }
        self._shape = (1, )
        self._value = {
            'min': -1.0,
            'max': 1.0,
            'dtype': float,
            'dinfo': 'float value',
        }

    def _to_agent_processor(self, reward: float) -> float:
        return reward

    def _from_agent_processor(self, reward: float) -> float:
        return reward

        # override
    def _details(self):
        return '\t'.join(self._reward_keys)

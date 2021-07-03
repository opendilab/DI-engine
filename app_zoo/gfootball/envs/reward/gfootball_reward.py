from collections import namedtuple

import numpy as np

from ding.envs.common import EnvElement


class GfootballReward(EnvElement):
    _name = "gfootballReward"
    _reward_keys = ['reward_value']
    Reward = namedtuple('Action', _reward_keys)

    MinReward = -1.0
    MaxReward = 1.0

    def _init(self, cfg) -> None:
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

    def _to_agent_processor(self, reward: float) -> np.array:
        return np.array([reward], dtype=float)

    def _from_agent_processor(self, reward: float) -> float:
        return reward

    def _details(self):
        return '\t'.join(self._reward_keys)

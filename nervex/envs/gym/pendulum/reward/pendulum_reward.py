import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class PendulumReward(EnvElement):
    def _init(self) -> None:
        self._default_val = 0.0
        self._shape = (1, )
        self._value = {
            'min': -1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2),
            'max': 0.0,
            'dtype': torch.FloatTensor,
            'dinfo': 'float value of reward, -(theta^2 + 0.1*theta_dot^2 + action^2)',
        }

    def _to_agent_processor(self, reward: float) -> float:
        return torch.FloatTensor([reward])

    def _from_agent_processor(self, reward: float) -> float:
        return reward

    def _details(self):
        return 'float value of reward, -(theta^2 + 0.1*theta_dot^2 + action^2)'

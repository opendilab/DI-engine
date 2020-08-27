import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class CartpoleObs(EnvElement):
    _name = "cartpoleObs"

    def _init(self):
        self._default_val = None
        self._shape = (4,)
        self._value = {
            'min': [-4.8, float("-inf"), -0.42, float("-inf")],
            'max': [4.8, float("inf"), 0.42, float("inf")],
            'dtype': np.ndarray,
            'dinfo': 'float value ndarray',
        }

    def _to_agent_processor(self, obs):
        return obs

    def _from_agent_processor(self, obs):
        return obs

    def _details(self):
        return '\t'.join(self._name)

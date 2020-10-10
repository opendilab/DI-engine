import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class CartpoleRawAction(EnvElement):
    _name = "cartpoleRawAction"

    def _init(self):
        self._default_val = None
        self._shape = (2, )
        self._value = {
            'min': 0,
            'max': 2,
            'dtype': torch.LongTensor,
            'dinfo': 'discreate int value, 0 is go left and 1 is go right'
        }

    def _to_agent_processor(self, action):
        return action

    def _from_agent_processor(self, action):
        return action

    # override
    def _details(self):
        return 'discreate int value, 0 is go left and 1 is go right'

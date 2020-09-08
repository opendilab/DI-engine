import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class PendulumRawAction(EnvElement):
    _name = "pendulumRawAction"

    def _init(self):
        self._default_val = None
        self._shape = (1, )
        self._value = {'min': -2.0, 'max': 2.0, 'dtype': float, 'dinfo': 'float value, the joint effort'}

    def _to_agent_processor(self, action):
        return action

    def _from_agent_processor(self, action, frame_skip):
        return [action] * frame_skip

    # override
    def _details(self):
        return 'float value from -2.0 to 2.0, the joint effort appied to pendulum'

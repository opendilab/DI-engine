import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class SumoObs(EnvElement):
    _name = "SumoObs"

    def _init(self):
        self._default_val = None
        # self.template = {
        #     'sumoObs':
        #     {
        #         'name': 'sumoObs',
        #         'shape' :(210, 160, 3), 
        #         'value' : {
        #             #TODO
        #             'min': 0,
        #             'max': 255,
        #             'dtype': np.ndarray,
        #             'dinfo': 'int value',
        #         },
        #         'env_value': '',
        #         'to_agent_processor': lambda x: x,
        #         'from_agent_processor': lambda x: x,
        #         'necessary': True,
        #     }
        # }
        self._shape = ('num of cars')
        self._value = 1
    
    def _to_agent_processor(self, obs):
        return obs

    def _from_agent_processor(self, obs):
        return obs

    # override
    def _details(self):
        return '\t'.join(self._name)
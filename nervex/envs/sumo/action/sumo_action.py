import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class SumoRawAction(EnvElement):
    _name = "SumoRawAction"

    def _init(self, cfg):
        self._cfg = cfg
        self._tls_green_action = cfg.tls_green_action
        self._tls_yellow_action = cfg.tls_yellow_action
        self._tls = self._tls_green_action.keys()
        self._shape = {k: len(v) for k, v in self._tls_green_action.items()}
        self._value = {
            k: {
                'min': 0,
                'max': self._shape[k] - 1,
                'dtype': int,
                'dinfo': 'int'
            }
            for k in self._tls_green_action.keys()
        }
        self._to_agent_processor = None

    def _from_agent_processor(self, data):
        print('data', data)
        data = tensor_to_list(data)
        raw_action = {k: {} for k in data.keys()}
        for k, v in data.items():
            action, last_action = v['action'], v['last_action']
            if last_action is None:
                yellow_phase = None
            else:
                yellow_phase = self._tls_yellow_action[k][last_action] if action != last_action else None
            raw_action[k]['yellow_phase'] = yellow_phase
            raw_action[k]['green_phase'] = self._tls_green_action[k][action]
        return raw_action

    # override
    def _details(self):
        return 'action dim: {}'.format(self._shape)

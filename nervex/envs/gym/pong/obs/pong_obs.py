import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement
from torchvision import transforms


class PongObs(EnvElement):
    _name = "pongObs"

    def _init(self, cfg):
        self._default_val = None
        self.template = {
            'pongObs': {
                'name': 'pongObs',
                'shape': (3, 210, 160),
                'value': {
                    'min': 0,
                    'max': 255,
                    'dtype': np.ndarray,
                    'dinfo': 'int value',
                },
                'env_value': '',
                'to_agent_processor': lambda x: x,
                'from_agent_processor': lambda x: x,
                'necessary': True,
            }
        }
        # 要改这边的shape还得把外面的cfg传进来，好麻烦
        self._shape = (3, 210, 160)
        self._value = {
            'min': 0,
            'max': 255,
            'dtype': torch.FloatTensor,
            'dinfo': 'float value tensor of shape (C, H, W)',
        }
        self._wrap_frame = cfg.get('wrap_frame', False)
        self._use_torch_wrap_frame = cfg.get('use_torch_wrap_frame', False)
        self._wrap_frame_height = cfg.get('wrap_frame_height', 84)
        self._wrap_frame_width = cfg.get('wrap_frame_width', 84)

        if self._wrap_frame:
            self._shape = (1, self._wrap_frame_height, self._wrap_frame_width)
            self._value = {
            'min': 0.0,
            'max': 1.0,
            'dtype': torch.FloatTensor,
            'dinfo': 'float value tensor of shape (C, H, W), C=1',
        }

    def _to_agent_processor(self, obs):
        return torch.from_numpy(obs).float()

    def _from_agent_processor(self, obs):
        return obs

    def _details(self):
        return '\t'.join(self._name)

import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class CartpoleReward(EnvElement):
    def _init(self) -> None:
        self._default_val = 0.0
        self._shape = (1, )
        self._value = {
            'min': 0.0,
            'max': 1.0,
            'dtype': float,
            'dinfo': 'float value of reward, 1.0 if cartpole hasnot fall'
        }

    def _to_agent_processor(self, reward: float) -> torch.tensor:
        return torch.FloatTensor([reward])

    def _from_agent_processor(self, reward: float) -> float:
        return reward

    def _details(self):
        return 'float value of reward, 1.0 if cartpole hasnot fall'

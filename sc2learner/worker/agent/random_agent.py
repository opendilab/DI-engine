from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.envs.spaces.pysc2_raw import PySC2RawAction
from .agent import BaseAgent


class RandomAgent(BaseAgent):
    '''Random agent.'''
    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, observation, eps=0):
        if (isinstance(self._action_space, MaskDiscrete) or isinstance(self._action_space, PySC2RawAction)):
            action_mask = observation[-1]
            return self._action_space.sample(np.nonzero(action_mask)[0])
        else:
            return self._action_space.sample()

    def reset(self):
        pass

    def value(self, obs):
        return 0

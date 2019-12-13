from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import queue
import threading

from absl import logging
import numpy as np

from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.envs.spaces.pysc2_raw import PySC2RawAction
from .agent import BaseAgent


def add_input(action_queue, n):
    while True:
        if action_queue.empty():
            cmds = input("Input Action ID: ")
            if not cmds.isdigit():
                print("Input should be an interger. Skipped.")
                continue
            action = int(cmds)
            if action >= 0 and action < n:
                action_queue.put(action)
            else:
                print("Invalid action. Skipped.")


class KeyboardAgent(BaseAgent):
    """A random agent for starcraft."""

    def __init__(self, action_space):
        super(KeyboardAgent, self).__init__()
        logging.set_verbosity(logging.ERROR)
        self._action_space = action_space
        self._action_queue = queue.Queue()
        self._cmd_thread = threading.Thread(
            target=add_input, args=(self._action_queue, action_space.n))
        self._cmd_thread.daemon = True
        self._cmd_thread.start()

    def act(self, observation, eps=0):
        time.sleep(0.1)
        if not self._action_queue.empty():
            action = self._action_queue.get()
            if (isinstance(self._action_space, MaskDiscrete) or
                    isinstance(self._action_space, PySC2RawAction)):
                action_mask = observation[-1]
                if action_mask[action] == 0:
                    print("Action not available. Availables: %s" %
                          np.nonzero(action_mask))
                    action = 0
            return action
        else:
            return 0

    def reset(self):
        pass

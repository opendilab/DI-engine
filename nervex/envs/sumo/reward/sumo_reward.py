import copy
import enum
from collections import namedtuple
from functools import partial
import numpy as np
import torch

from nervex.torch_utils import to_tensor, tensor_to_list
from nervex.envs.common import EnvElement


class SumoReward(EnvElement):
    _name = "sumoReward"

    def wait_time(self, wait_time_reward):
        return wait_time_reward

    def queue_len(self, queue_len):
        return -queue_len

    def delay_time(self, collect_delay_time):
        return collect_delay_time

    def _init(self, reward_types) -> None:
        self.reward_types = reward_types
        self._reward_keys = ['wait_time', 'queue_len', 'delay_time']
        self._shape = {k: (1, ) for k in self._reward_keys}
        self._value = {'wait_time': {}, 'queue_len': {}, 'delay_time': {}}
        self.reward_func_dict = {'wait_time': self.wait_time, 'queue_len': self.queue_len, 'delay_time': self.delay_time}

    def _to_agent_processor(self, reward_types, wait_time_reward, queue_len, collect_delay_time):
        reward = {}
        for t in self.reward_types:
            if t == 'wait_time':
                reward[t] = wait_time_reward
            elif t == 'queue_len':
                reward[t] = - queue_len
            elif t == 'delay_time':
                reward[t] = collect_delay_time
        #     reward[t] = self.reward_func_dict[t]()
        if len(reward) == 1:
            return list(reward.values())[0]
        else:
            return reward

    def _from_agent_processor(self):
        return None

        # override
    def _details(self):
        return '\t'.join(self._reward_keys)

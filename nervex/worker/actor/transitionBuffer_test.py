
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
from easydict import EasyDict

class TransitionBuffer(object):

    def __init__(self, env_num: int, max_episode_len: int, enforce_padding: Optional[bool] = False, null_transition: Optional[Dict] = {}):
        self._env_num = env_num
        self._buffer = {env_id: [] for env_id in range(env_num)}
        self._left_flags = [False for env_id in range(env_num)]
        self.max_episode_len = max_episode_len
        self.enforce_padding = enforce_padding
        self.null_transition = null_transition

    def append(self, env_id: int, transition: dict):
        assert env_id < self._env_num
        self._buffer[env_id].append(transition)

    def get_episode(self, env_id: int) -> List[dict]:
        stored_epi = self._buffer[env_id]
        left_flag = self._left_flags[env_id]
        ret_epi = None

        if stored_epi[-1].done: # episode finishes
            if left_flag: # transitions left from last time, shift the episode to pad
                ret_epi = copy.deepcopy(stored_epi[-self.max_episode_len:])
            else: # can't shift to pad
                ret_epi = copy.deepcopy(stored_epi)
                if self.enforce_padding: 
                    ret_epi += [self.null_transition for _ in range(self.max_episode_len - len(stored_epi))]
            self._buffer[env_id] = []
            self._left_flags[env_id] = False
            return ret_epi
        elif len(stored_epi) == (1 + left_flag)*self.max_episode_len: # enough transitions
            ret_epi = copy.deepcopy(stored_epi[-self.max_episode_len:])
            self._buffer[env_id] = ret_epi
            self._left_flags[env_id] = True
            return ret_epi

        return None                

    def get_buffer(self) -> Dict[int, List]:
        return self._buffer



middle_transition = EasyDict({'done': False})
last_transition = EasyDict({'done': True})
tb = TransitionBuffer(env_num=2, max_episode_len=3, enforce_padding=True)
env_id = 0
for i in range(10):
    tb.append(env_id, middle_transition)
    print(tb.get_episode(env_id))

tb.append(env_id, last_transition)
print(tb.get_episode(env_id))
tb.append(env_id, last_transition)
print(tb.get_episode(env_id))

for i in range(10):
    tb.append(env_id, middle_transition)
    print(tb.get_episode(env_id))
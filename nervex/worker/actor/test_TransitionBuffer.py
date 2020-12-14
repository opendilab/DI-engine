
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
from easydict import EasyDict
import pytest

@pytest.mark.unittest
class TestTransitionBuffer(object):

    def __init__(self, env_num: int, max_traj_len: int, enforce_padding: Optional[bool] = False, null_transition: Optional[Dict] = {}):
        self._env_num = env_num
        self._buffer = {env_id: [] for env_id in range(env_num)}
        self._left_flags = [False for env_id in range(env_num)]
        self.max_traj_len = max_traj_len
        self.enforce_padding = enforce_padding
        self.null_transition = null_transition

    def append(self, env_id: int, transition: dict):
        assert env_id < self._env_num
        self._buffer[env_id].append(transition)

    def get_traj(self, env_id: int) -> List[dict]:
        stored_epi = self._buffer[env_id]
        left_flag = self._left_flags[env_id]
        ret_epi = None
        assert len(stored_epi) > 0, "request traj before insert a new transition!"

        if stored_epi[-1].done: # episode finishes
            if left_flag: # transitions left from last time, shift the episode to pad
                ret_epi = copy.deepcopy(stored_epi[-self.max_traj_len:])
            else: # can't shift to pad
                ret_epi = copy.deepcopy(stored_epi)
                if self.enforce_padding: 
                    ret_epi += [self.null_transition for _ in range(self.max_traj_len - len(stored_epi))]
            self._buffer[env_id] = []
            self._left_flags[env_id] = False
            return ret_epi
        elif len(stored_epi) >= (1 + left_flag)*self.max_traj_len: # enough transitions
            ret_epi = copy.deepcopy(stored_epi[-self.max_traj_len:])
            self._buffer[env_id] = ret_epi
            self._left_flags[env_id] = True
            return ret_epi

        return None        

    @property
    def buffer(self) -> Dict[int, List]:
        return self._buffer

    def clear(self) -> None:
        self._buffer = {env_id: [] for env_id in range(env_num)}
        self._left_flags = [False for env_id in range(self._env_num)]    
  


if __name__ == "__main__":
    middle_transition = EasyDict({'done': False})
    last_transition = EasyDict({'done': True})
    max_traj_len = 3
    env_num = 2
    env_id = 0

    tb = TestTransitionBuffer(env_num=env_num, max_traj_len=max_traj_len, enforce_padding=False)
    # Test 1: achive max len and return traj
    for idx in range(9):
        tb.append(env_id, middle_transition)
        if (idx+1)%max_traj_len == 0: 
            assert len(tb.get_traj(env_id)) == max_traj_len
        else:
            assert tb.get_traj(env_id) == None
    # Test 2: done and return
    tb.clear()
    tb.append(env_id, middle_transition)
    tb.get_traj(env_id)
    tb.append(env_id, last_transition)

    assert len(tb.get_traj(env_id)) == 2

    # Test 3: done and shift the buffer to return
    tb.clear()
    for idx in range(3):
        tb.append(env_id, middle_transition)
        tb.get_traj(env_id)
    tb.append(env_id, last_transition)
    assert len(tb.get_traj(env_id)) == 3

    # Test 4: done and complete the return with null
    tb = TestTransitionBuffer(env_num=env_num, max_traj_len=max_traj_len, enforce_padding=True)
    tb.append(env_id, last_transition)
    new_traj = tb.get_traj(env_id)
    assert len(new_traj) == 3
    assert new_traj[-1] == {}
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
from easydict import EasyDict
import pytest


class TestTransitionBuffer(object):

    def test_naive(self):
        middle_transition = EasyDict({'done': False})
        last_transition = EasyDict({'done': True})
        max_traj_len = 3
        env_num = 2
        env_id = 0

        tb = TransitionBuffer(env_num=env_num, max_traj_len=max_traj_len, enforce_padding=False)
        # Test 1: achive max len and return traj
        for idx in range(9):
            tb.append(env_id, middle_transition)
            if (idx + 1) % max_traj_len == 0:
                assert len(tb.get_traj(env_id)) == max_traj_len
            else:
                assert tb.get_traj(env_id) is None
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
        tb = TransitionBuffer(env_num=env_num, max_traj_len=max_traj_len, enforce_padding=True)
        tb.append(env_id, last_transition)
        new_traj = tb.get_traj(env_id)
        assert len(new_traj) == 3
        assert new_traj[-1] == {}

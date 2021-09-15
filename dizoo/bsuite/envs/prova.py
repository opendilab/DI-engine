from time import time
import pytest
import numpy as np
from easydict import EasyDict
from dizoo.bsuite.envs import BSuiteEnv
import bsuite
from bsuite.utils import gym_wrapper


raw_env = bsuite.load_from_id(bsuite_id='memory_len/0')
memory_len_env = gym_wrapper.GymFromDMEnv(raw_env)
s = memory_len_env.reset()
print(s)
for _ in range(20):
    s, r, d, i = memory_len_env.step(0)
    print(s, r, d)
    if d:
        print('erg')


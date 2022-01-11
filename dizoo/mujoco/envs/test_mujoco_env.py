import os
import pytest
import numpy as np
from easydict import EasyDict

from ding.utils import set_pkg_seed
from dizoo.mujoco.envs import MujocoEnv


@pytest.mark.envtest
@pytest.mark.parametrize('delay_reward_step', [1, 10])
def test_mujoco_env(delay_reward_step):
    set_pkg_seed(1234, use_cuda=False)
    env = MujocoEnv(EasyDict({'env_id': 'Ant-v3', 'use_act_scale': False, 'delay_reward_step': delay_reward_step}))
    env.seed(1234)
    env.reset()
    action_dim = env.info().act_space.shape
    for _ in range(25):
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        print(_, timestep.reward)
        assert timestep.reward.shape == (1, ), timestep.reward.shape

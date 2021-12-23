import pytest
import numpy as np
from easydict import EasyDict
from dizoo.mujoco.envs import MujocoEnv


@pytest.mark.envtest
@pytest.mark.parametrize('delay_reward_step', [0, 10])
def test_mujoco_env(delay_reward_step):
    env = MujocoEnv(EasyDict({'env_id': 'Ant-v3', 'use_act_scale': False, 'delay_reward_step': delay_reward_step}))
    env.reset()
    action_dim = env.info().act_space.shape
    for _ in range(25):
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        print(_, timestep.reward)
        assert timestep.reward.shape == (1, )

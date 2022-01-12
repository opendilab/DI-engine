import os
import pytest
import numpy as np
from easydict import EasyDict

from ding.utils import set_pkg_seed
from dizoo.mujoco.envs import MujocoEnv


@pytest.mark.envtest
@pytest.mark.parametrize('delay_reward_step', [1, 10])
def test_mujoco_env_delay_reward(delay_reward_step):
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


@pytest.mark.envtest
def test_mujoco_env_final_eval_reward():
    set_pkg_seed(1234, use_cuda=False)
    env = MujocoEnv(EasyDict({'env_id': 'Ant-v3', 'use_act_scale': False, 'delay_reward_step': 4}))
    env.seed(1234)
    env.reset()
    action_dim = env.info().act_space.shape
    final_eval_reward = np.array([0.], dtype=np.float32)
    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        final_eval_reward += timestep.reward
        # print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print("{}({}), {}({})".format(timestep.info['final_eval_reward'], type(timestep.info['final_eval_reward']),
                                          final_eval_reward, type(final_eval_reward)))
            # timestep.reward and the cumulative reward in wrapper FinalEvalReward are not the same.
            assert abs(timestep.info['final_eval_reward'].item() - final_eval_reward.item()) / \
                abs(timestep.info['final_eval_reward'].item()) < 1e-5
            break

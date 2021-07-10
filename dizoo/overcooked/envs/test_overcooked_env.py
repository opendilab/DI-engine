from time import time
from numpy.core.shape_base import stack
import pytest
import numpy as np
from dizoo.overcooked.envs import OvercookEnv


@pytest.mark.unittest
class TestOvercooked:

    def test_overcook(self):
        stack_obs = True
        num_agent = 2
        sum_rew = 0.0
        env = OvercookEnv({'stack_obs': stack_obs})
        # print(env.info())
        obs = env.reset()
        for k, v in obs.items():
            # print("obs space is", env.info().obs_space.shape)
            if stack_obs:
                assert v.shape == env.info().obs_space.shape[k]
            else:
                assert len(v) == len(env.info().obs_space.shape[k])
        for _ in range(env._horizon):
            action = np.random.randint(0, 6, (num_agent, ))
            # print("action is:", action)
            timestep = env.step(action)
            obs = timestep.obs
            # print("reward = ", timestep.reward)
            # print("done = ", timestep.done)
            # print("timestep = ", timestep)
            for k, v in obs.items():
                if stack_obs:
                    assert v.shape == env.info().obs_space.shape[k]
                else:
                    assert len(v) == len(env.info().obs_space.shape[k])
            # assert isinstance(timestep, tuple), timestep
        # print("Test done for {} steps".for
        assert timestep.done
        # print("final reward=", timestep.info['final_eval_reward'])
        sum_rew += timestep.info['final_eval_reward'][0]


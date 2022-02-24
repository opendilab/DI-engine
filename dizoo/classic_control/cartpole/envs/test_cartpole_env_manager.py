import pytest
import numpy as np
from ding.envs import BaseEnvManager
from dizoo.classic_control.cartpole.envs import CartPoleEnv


@pytest.mark.envtest
class TestCartPoleEnv:

    def test_naive(self):
        env_num = 8
        env = BaseEnvManager([lambda: CartPoleEnv({}) for _ in range(env_num)], BaseEnvManager.default_config())
        env.seed(314, dynamic_seed=False)
        env.launch()
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            for i in range(10):
                obs = env.ready_obs
                assert len(obs) == env_num
                random_action = {i: np.array([env.action_space.sample()]) for i in range(env_num)}
                timesteps = env.step(random_action)
                # print(timesteps)
                assert isinstance(timesteps, dict)
                # test one of timesteps
                timestep = timesteps[0]
                assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs.shape == (4, )
                assert timestep.reward.shape == (1, )
                assert timestep.reward >= env.reward_space.low
                assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

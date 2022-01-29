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
        act_val = env.env_info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            for i in range(10):
                obs = env.ready_obs
                assert len(obs) == env_num
                random_action = {i: np.random.randint(min_val, max_val, size=(1, )) for i in range(env_num)}
                timesteps = env.step(random_action)
                # print(timesteps)
                assert isinstance(timesteps, dict)
                # test one of timesteps
                timestep = timesteps[0]
                assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs.shape == (4, )
                assert timestep.reward.shape == (1, )
                assert timestep.reward >= env.env_info().rew_space.value['min']
                assert timestep.reward <= env.env_info().rew_space.value['max']
        print(env.env_info())
        env.close()

import pytest
import numpy as np
from easydict import EasyDict
from dizoo.gym_anytrading.envs import StocksEnv


@pytest.mark.envtest
class TestStocksEnv:

    def test_naive(self):
        env = StocksEnv(EasyDict({"env_id": 'stocks-v0', "eps_length": 300,\
            "window_size": 20, "train_range": None, "test_range": None, "stocks_data_filename": 'STOCKS_GOOGL'}))
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (62, )
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            print('=' * 60)
            for i in range(10):
                # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
                # can generate legal random action.
                if i < 5:
                    random_action = np.array([env.action_space.sample()])
                else:
                    random_action = env.random_action()
                timestep = env.step(random_action)
                print(timestep)
                assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs.shape == (62, )
                assert timestep.reward.shape == (1, )
                assert timestep.reward >= env.reward_space.low
                assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

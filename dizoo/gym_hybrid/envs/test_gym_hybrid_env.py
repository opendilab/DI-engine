import pytest
import numpy as np
from easydict import EasyDict

from dizoo.gym_hybrid.envs import GymHybridEnv


@pytest.mark.envtest
class TestGymHybridEnv:

    def test_naive(self):
        env = GymHybridEnv(EasyDict({'env_id': 'Moving-v0', 'act_scale': False}))
        env.enable_save_replay('./video')
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (10, )
        for i in range(200):
            random_action = env.random_action()
            print('random_action', random_action)
            timestep = env.step(random_action)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (10, )
            assert timestep.reward.shape == (1, )
            assert timestep.info['action_args_mask'].shape == (3, 2)
            if timestep.done:
                print('reset env')
                env.reset()
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

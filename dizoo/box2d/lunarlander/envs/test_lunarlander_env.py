from time import time
import pytest
import numpy as np
from easydict import EasyDict
from dizoo.box2d.lunarlander.envs import LunarLanderEnv


@pytest.mark.envtest
@pytest.mark.parametrize(
    'cfg', [
        EasyDict({
            'env_id': 'LunarLander-v2',
            'act_scale': False
        }),
        EasyDict({
            'env_id': 'LunarLanderContinuous-v2',
            'act_scale': True
        })
    ]
)
class TestLunarLanderEnvEnv:

    def test_naive(self, cfg):
        env = LunarLanderEnv(cfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (8, )
        for i in range(10):
            random_action = env.random_action()
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (8, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
            # assert isinstance(timestep, tuple)
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

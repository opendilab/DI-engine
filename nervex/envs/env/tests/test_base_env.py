import random

import numpy as np
import pytest
import torch

from nervex.envs.env.base_env import NervexEnvWrapper

@pytest.mark.unittest
class TestNervexEnvWrapper:

    def test_naive(self):
        env = gym.make('PongNoFrameskip-v4')
        nervex_env = NervexEnvWrapper(env)
        info = nervex_env.info()
        assert info is not None
        l1 = nervex_env.create_actor_env_cfg()
        assert isinstance(l1, list)
        l1 = nervex_env.create_evaluator_env_cfg()
        assert isinstance(l1, list)
        obs = nervex_env.reset()
        assert isinstance(obs, np.ndarray)


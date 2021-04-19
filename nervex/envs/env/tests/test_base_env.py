import random
import gym
import numpy as np
import pytest
import torch
from easydict import EasyDict

from nervex.envs.env.nervex_env_wrapper import NervexEnvWrapper


@pytest.mark.unittest
class TestNervexEnvWrapper:

    def test_naive(self):
        env = gym.make('PongNoFrameskip-v4')
        nervex_env = NervexEnvWrapper(env)
        info = nervex_env.info()
        assert info is not None
        cfg = EasyDict(dict(
            actor_env_num=16,
            evaluator_env_num=3,
            is_train=True,
        ))
        l1 = nervex_env.create_actor_env_cfg(cfg)
        assert isinstance(l1, list)
        l1 = nervex_env.create_evaluator_env_cfg(cfg)
        assert isinstance(l1, list)
        obs = nervex_env.reset()
        assert isinstance(obs, np.ndarray)

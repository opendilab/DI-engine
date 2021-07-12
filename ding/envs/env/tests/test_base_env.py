import random
import gym
import numpy as np
import pytest
import torch
from easydict import EasyDict

from ding.envs.env.ding_env_wrapper import DingEnvWrapper


@pytest.mark.unittest
class TestDingEnvWrapper:

    def test_naive(self):
        env = gym.make('CartPole-v0')
        ding_env = DingEnvWrapper(env)
        info = ding_env.info()
        assert info is not None
        cfg = EasyDict(dict(
            collector_env_num=16,
            evaluator_env_num=3,
            is_train=True,
        ))
        l1 = ding_env.create_collector_env_cfg(cfg)
        assert isinstance(l1, list)
        l1 = ding_env.create_evaluator_env_cfg(cfg)
        assert isinstance(l1, list)
        obs = ding_env.reset()
        assert isinstance(obs, np.ndarray)

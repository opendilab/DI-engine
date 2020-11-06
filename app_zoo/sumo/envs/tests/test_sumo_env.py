import os
import random

import pytest
import torch
import yaml
from easydict import EasyDict

from app_zoo.sumo.envs import SumoWJ3Env


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../sumo_env_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg.env


@pytest.mark.envtest
class TestSumoWJ3Env:

    def get_random_action(self, action_dim):
        action = []
        for k, v in action_dim.items():
            action.append(random.choice(list(range(v))))
        action = [torch.LongTensor([v]) for v in action]
        return action

    def test_naive(self, setup_config):
        env = SumoWJ3Env(setup_config)
        print(env)
        obs = env.reset()
        for i in range(10):
            action = self.get_random_action(env.info().act_space.shape)
            timestep = env.step(action)
            print(timestep.reward)
            print('step {} with action {}'.format(i, action))
        print('end')

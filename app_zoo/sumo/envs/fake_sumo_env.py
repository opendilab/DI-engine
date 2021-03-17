import os
from collections import namedtuple

import torch
import yaml
from easydict import EasyDict

from nervex.utils import deep_merge_dicts, ENV_REGISTRY
from .sumo_env import SumoWJ3Env


def build_config(user_config):
    """Aggregate a general config"""
    with open(os.path.join(os.path.dirname(__file__), 'sumo_env_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    default_config = cfg.env
    return deep_merge_dicts(default_config, user_config)


@ENV_REGISTRY.register('sumo_wj3_fake')
class FakeSumoWJ3Env(SumoWJ3Env):
    timestep = namedtuple('SumoTimestep', ['obs', 'reward', 'done', 'info'])
    info_template = namedtuple('SumoWJ3EnvInfo', ['obs_space', 'act_space', 'rew_space', 'agent_num'])

    def __init__(self, cfg: dict) -> None:
        cfg = build_config(cfg)
        super().__init__(cfg)
        self.tls = ['htxdj_wjj', 'haxl_wjj', 'haxl_htxdj']
        self.count = 0

    def reset(self):
        return torch.randn(380)

    def close(self):
        pass

    def seed(self, seed):
        pass

    def step(self, action: list) -> 'FakeSumoWJ3Env.timestep':
        obs = torch.randn(380)
        reward = -torch.randn(1)
        done = self.count >= 200
        info = {}
        if done:
            info['final_eval_reward'] = -torch.randn(1).item()
        self.count += 1
        return FakeSumoWJ3Env.timestep(obs, reward, done, info)

    def __repr__(self):
        return 'FakeSumoWJ3Env'

import copy
import os
from collections import namedtuple
import sys
from typing import List, Any
import numpy as np
import torch
import yaml
from easydict import EasyDict
from nervex.utils import override, merge_dicts, pretty_print, read_config
from .sumo_env import SumoWJ3Env

import time
from functools import reduce


def build_config(user_config):
    """Aggregate a general config"""
    with open(os.path.join(os.path.dirname(__file__), 'sumo_env_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    default_config = cfg.env
    return merge_dicts(default_config, user_config)


class FakeSumoWJ3Env(SumoWJ3Env):
    timestep = namedtuple('SumoTimestep', ['obs', 'reward', 'done', 'info'])
    info_template = namedtuple('SumoWJ3EnvInfo', ['obs_space', 'act_space', 'rew_space', 'agent_num'])

    def __init__(self, cfg: dict) -> None:
        cfg = build_config(cfg)
        super().__init__(cfg)
        self.tls = ['htxdj_wjj', 'haxl_wjj', 'haxl_htxdj']

    def reset(self):
        return torch.randn(380)

    def close(self):
        pass

    def seed(self):
        pass

    def step(self, action: list) -> 'FakeSumoWJ3Env.timestep':
        obs = torch.randn(380)
        reward = {k: -torch.rand(1) for k in ['queue_len', 'wait_time', 'delay_time']}
        return FakeSumoWJ3Env.timestep(obs, reward, False, {'cum_reward': 0})

    def __repr__(self):
        return 'FakeSumoWJ3Env'


SumoTimestep = FakeSumoWJ3Env.timestep

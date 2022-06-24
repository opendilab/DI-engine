import pytest
import torch
import os
import yaml
from easydict import EasyDict
# from dizoo.gfootball.envs.gfootball_env import GfootballEnv
# from ding.utils import read_config


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../iql_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg

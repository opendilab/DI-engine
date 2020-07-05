import pytest
import os
import yaml
from easydict import EasyDict
import pysc2.env.sc2_env as sc2_env


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../alphastar_env_default_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg

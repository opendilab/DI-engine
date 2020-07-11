import pytest
import torch
import os
import yaml
from easydict import EasyDict
from sc2learner.envs import AlphaStarEnv


@pytest.fixture(scope='module')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../actor_critic_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


def is_differentiable(loss, model):
    assert isinstance(loss, torch.Tensor)
    assert isinstance(model, torch.nn.Module)
    for p in model.parameters():
        assert p.grad is None
    loss.backward()
    for k, p in model.named_parameters():
        assert isinstance(p.grad, torch.Tensor), k


@pytest.fixture(scope='module')
def setup_env_info():
    env = AlphaStarEnv(cfg={})
    return env.info()

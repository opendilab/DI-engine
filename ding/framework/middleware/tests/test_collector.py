import pytest
import torch
import copy
from unittest.mock import patch
from ding.framework import Context
from ding.framework.middleware.functional.collector import inferencer, rolloutor
from ding.framework.middleware.tests.mock_for_test import MockPolicy, MockEnv, CONFIG
    

@pytest.mark.lxl
def test_inferencer():
    cfg = copy.deepcopy(CONFIG)
    ctx = Context(CONFIG.ctx)
    with patch("ding.policy.Policy",  MockPolicy):
        with patch("ding.envs.BaseEnvManagerV2",  MockEnv):
            policy = MockPolicy()
            env = MockEnv()
            inferencer(cfg, policy, env)(ctx)
    assert isinstance(ctx.inference_output, dict)
    assert ctx.inference_output[0] == {'action': torch.Tensor([0.])}
    assert ctx.inference_output[1] == {'action': torch.Tensor([4.])}


@pytest.mark.lxl
def test_rolloutor():
    cfg = copy.deepcopy(CONFIG)
    ctx = Context(CONFIG.ctx)
    transitions = [[], []]
    with patch("ding.policy.Policy",  MockPolicy):
        with patch("ding.envs.BaseEnvManagerV2",  MockEnv):
            policy = MockPolicy()
            env = MockEnv()
            for _ in range(10):
                inferencer(cfg, policy, env)(ctx)
                rolloutor(cfg, policy, env, transitions)(ctx)
    assert ctx.env_episode == 20           # 10 * env_num
    assert ctx.env_step == 20              # 10 * env_num


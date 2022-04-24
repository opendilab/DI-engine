import pytest
import torch
import copy
from unittest.mock import patch
from ding.framework import Context, task
from ding.framework.middleware.functional.evaluator import interaction_evaluator
from ding.framework.middleware.tests.mock_for_test import MockPolicy, MockEnv, CONFIG


@pytest.mark.lxl
def test_interaction_evaluator():
    cfg = copy.deepcopy(CONFIG)
    ctx = Context(CONFIG.ctx)
    with patch("ding.policy.Policy",  MockPolicy):
        with patch("ding.envs.BaseEnvManagerV2",  MockEnv):
            with task.start():
                policy = MockPolicy()
                env = MockEnv()
                for _ in range(100):
                    ctx.train_iter += 1
                    interaction_evaluator(cfg, policy, env)(ctx)
    print("ctx.last_eval_iter:", ctx.last_eval_iter)
    print("ctx.train_iter:", ctx.train_iter)

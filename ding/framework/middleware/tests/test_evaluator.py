import pytest
import torch
import copy
from unittest.mock import patch
from ding.framework import OnlineRLContext, task
from ding.framework.middleware import interaction_evaluator
from ding.framework.middleware.tests import MockPolicy, MockEnv, CONFIG


@pytest.mark.unittest
def test_interaction_evaluator():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        with task.start():
            policy = MockPolicy()
            env = MockEnv()
            for i in range(30):
                ctx.train_iter += 1
                interaction_evaluator(cfg, policy, env)(ctx)
                # interaction_evaluator will run every 10 train_iter in the test
                assert ctx.last_eval_iter == i // 10 * 10 + 1
                # the reward will increase 1.0 each step.
                # there are 2 env_num and 5 episodes in the test.
                # so when interaction_evaluator runs the first time, reward is [[1, 2, 3], [2, 3]] and the avg = 2.2
                # the second time, reward is [[4, 5, 6], [5, 6]] . . .
                assert ctx.eval_value == 2.2 + i // 10 * 3.0

#unittest for priority_calculator

import unittest
import pytest
import numpy as np
from unittest.mock import Mock, patch
from ding.framework import OnlineRLContext, OfflineRLContext
from ding.framework import task, Parallel
from ding.framework.middleware.functional import priority_calculator


class MockPolicy(Mock):

    def priority_fun(self, data):
        return np.random.rand(len(data))


@pytest.mark.unittest
def test_priority_calculator():
    policy = MockPolicy()
    ctx = OnlineRLContext()
    ctx.trajectories = [
        {
            'obs': np.random.rand(2, 2),
            'next_obs': np.random.rand(2, 2),
            'reward': np.random.rand(1),
            'info': {}
        } for _ in range(10)
    ]
    priority_calculator_middleware = priority_calculator(priority_calculation_fn=policy.priority_fun)
    priority_calculator_middleware(ctx)
    assert len(ctx.trajectories) == 10
    assert all([isinstance(traj['priority'], float) for traj in ctx.trajectories])

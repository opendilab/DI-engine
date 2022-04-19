import pytest
import math
import time

from easydict import EasyDict
from ding.model import DQN
from ding.policy.dqn import Policy
from ding.policy.policy_factory import get_random_policy
from ding.framework import task, Context
from ding.framework import Parallel
from ding.framework.middleware.functional.collector import episode_collector
from ding.data import DequeBuffer



@pytest.mark.unittest
def test_episode_collector():
    cfg = Policy.default_config()
    cfg.seed = 0
    random_policy = get_random_policy()
    buffer = DequeBuffer(size=1000)
    policy = Policy(cfg)

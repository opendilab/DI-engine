import pytest

from ding.envs.env import check_reset, check_step, check_obs_deepcopy, \
    demonstrate_correct_procudure
from ding.envs.env.tests import DemoEnv


@pytest.mark.unittest
def test_an_implemented_env():
    demo_env = DemoEnv({})
    check_reset(demo_env)
    check_step(demo_env)
    check_obs_deepcopy(demo_env)
    demonstrate_correct_procudure(DemoEnv)

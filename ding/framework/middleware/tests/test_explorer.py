import pytest
import copy
from ding.framework import OnlineRLContext
from ding.framework.middleware import eps_greedy_handler, eps_greedy_masker
from ding.framework.middleware.tests import MockPolicy, MockEnv, CONFIG


@pytest.mark.unittest
def test_eps_greedy_handler():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()

    ctx.env_step = 0
    next(eps_greedy_handler(cfg)(ctx))
    assert ctx.collect_kwargs['eps'] == 0.95

    ctx.env_step = 1000000
    next(eps_greedy_handler(cfg)(ctx))
    assert ctx.collect_kwargs['eps'] == 0.1


@pytest.mark.unittest
def test_eps_greedy_masker():
    ctx = OnlineRLContext()
    for _ in range(10):
        eps_greedy_masker()(ctx)
    assert ctx.collect_kwargs['eps'] == -1

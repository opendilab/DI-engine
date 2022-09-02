import pytest
import pickle
import numpy as np
from ding.framework import Context, OnlineRLContext, OfflineRLContext
from dataclasses import dataclass


@dataclass
class MockContext(Context):
    hello: str = "world"
    keep_me: int = 0
    not_keep_me: int = 0


@pytest.mark.unittest
def test_pickable():
    ctx = MockContext()
    ctx.keep("keep_me")
    _ctx = pickle.loads(pickle.dumps(ctx))
    assert _ctx.hello == "world"

    ctx.keep_me += 1
    ctx.not_keep_me += 1

    _ctx = ctx.renew()
    assert _ctx.keep_me == 1
    assert _ctx.not_keep_me == 0


@pytest.mark.unittest
def test_online():
    ctx = OnlineRLContext()
    assert ctx.env_step == 0
    assert ctx.eval_value == -np.inf

    ctx.env_step += 1
    ctx.eval_value = 1
    assert ctx.env_step == 1
    assert ctx.eval_value == 1

    _ctx = ctx.renew()
    assert _ctx.env_step == 1
    assert _ctx.eval_value == -np.inf


@pytest.mark.unittest
def test_offline():
    ctx = OfflineRLContext()
    assert ctx.train_iter == 0
    assert ctx.eval_value == -np.inf

    ctx.train_iter += 1
    ctx.eval_value = 1
    assert ctx.train_iter == 1
    assert ctx.eval_value == 1

    _ctx = ctx.renew()
    assert _ctx.train_iter == 1
    assert _ctx.eval_value == -np.inf

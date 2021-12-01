import pytest
from ding.framework import Context
import pickle


@pytest.mark.unittest
def test_pickable():
    ctx = Context(hello="world", keep_me=True)
    ctx.keep("keep_me")
    _ctx = pickle.loads(pickle.dumps(ctx))
    assert _ctx.hello == "world"

    _ctx = ctx.renew()
    assert _ctx.keep_me

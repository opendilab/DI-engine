import pytest
from ding.framework import Context
import pickle


@pytest.mark.unittest
def test_pickable():
    ctx = Context(hello="world")
    ctx.keep("any")
    _ctx = pickle.loads(pickle.dumps(ctx))
    assert _ctx.hello == "world"
    assert len(_ctx._hooks_after_renew) < len(ctx._hooks_after_renew)

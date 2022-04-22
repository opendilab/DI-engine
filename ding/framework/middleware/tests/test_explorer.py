import pytest
import copy
from ding.framework import Context
from ding.framework.middleware.functional.explorer import eps_greedy_handler
from ding.framework.middleware.tests.mock_for_test import MockPolicy, MockEnv, CONFIG
    

@pytest.mark.lxl
def test_eps_greedy_handler():
    cfg = copy.deepcopy(CONFIG)
    ctx = Context(CONFIG.ctx)
    for _ in range(100):
        eps_greedy_handler(cfg)(ctx)
    print("Done")



import pytest
from copy import deepcopy
from ding.entry import parallel_pipeline
from dizoo.classic_control.cartpole.config.parallel.cartpole_dqn_config import main_config, create_config,\
    system_config


# @pytest.mark.unittest
@pytest.mark.execution_timeout(120.0, method='thread')
def test_dqn():
    config = tuple([deepcopy(main_config), deepcopy(create_config), deepcopy(system_config)])
    config[0].env.stop_value = 9
    try:
        parallel_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

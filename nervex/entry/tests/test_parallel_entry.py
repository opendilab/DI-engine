import pytest
from copy import deepcopy
from nervex.entry import parallel_pipeline
from app_zoo.classic_control.cartpole.entry.parallel.cartpole_dqn_default_config import main_config


@pytest.mark.unittest
@pytest.mark.execution_timeout(180.0, method='thread')
def test_dqn():
    config = deepcopy(main_config)
    config.coordinator.commander.actor_cfg.env_kwargs.eval_stop_val = 15
    config.coordinator.commander.eval_interval = 5
    try:
        parallel_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

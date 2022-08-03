import pytest
from copy import deepcopy
from ding.entry.serial_entry_for_anytrading import serial_pipeline_for_anytrading
from dizoo.gym_anytrading.config import stocks_dqn_config, stocks_dqn_create_config

@pytest.mark.platformtest
@pytest.mark.unittest
def test_stocks_dqn():
    config = [deepcopy(stocks_dqn_config), deepcopy(stocks_dqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'stocks_dqn_unittest'
    try:
        serial_pipeline_for_anytrading(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

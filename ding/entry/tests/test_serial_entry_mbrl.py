import pytest
from copy import deepcopy
from ding.entry.serial_entry_mbrl import serial_pipeline_dyna, serial_pipeline_dream

from dizoo.classic_control.pendulum.config.mbrl.pendulum_sac_mbpo_config \
    import main_config as pendulum_sac_mbpo_main_config,\
    create_config as pendulum_sac_mbpo_create_config

from dizoo.classic_control.pendulum.config.mbrl.pendulum_mbsac_mbpo_config \
    import main_config as pendulum_mbsac_mbpo_main_config,\
    create_config as pendulum_mbsac_mbpo_create_config


@pytest.mark.unittest
def test_sac_mbpo_dyna():
    config = [deepcopy(pendulum_sac_mbpo_main_config), deepcopy(pendulum_sac_mbpo_create_config)]
    try:
        serial_pipeline_dyna(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac_mbpo_dream():
    config = [deepcopy(pendulum_mbsac_mbpo_main_config), deepcopy(pendulum_mbsac_mbpo_create_config)]
    try:
        serial_pipeline_dream(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

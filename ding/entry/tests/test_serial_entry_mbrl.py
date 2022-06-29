import pytest
from copy import deepcopy
from ding.entry.serial_entry_mbrl import serial_pipeline_dyna, serial_pipeline_dream

from dizoo.classic_control.pendulum.config.mbrl.pendulum_sac_mbpo_config \
    import main_config as pendulum_sac_mbpo_main_config,\
    create_config as pendulum_sac_mbpo_create_config

from dizoo.classic_control.pendulum.config.mbrl.pendulum_mbsac_mbpo_config \
    import main_config as pendulum_mbsac_mbpo_main_config,\
    create_config as pendulum_mbsac_mbpo_create_config

from dizoo.classic_control.pendulum.config.mbrl.pendulum_stevesac_mbpo_config \
    import main_config as pendulum_stevesac_mbpo_main_config,\
    create_config as pendulum_stevesac_mbpo_create_config


@pytest.mark.unittest
def test_dyna():
    config = [deepcopy(pendulum_sac_mbpo_main_config), deepcopy(pendulum_sac_mbpo_create_config)]
    config[0].world_model.model.max_epochs_since_update = 0
    try:
        serial_pipeline_dyna(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_dream():
    configs = [
        [deepcopy(pendulum_mbsac_mbpo_main_config),
         deepcopy(pendulum_mbsac_mbpo_create_config)],
        [deepcopy(pendulum_stevesac_mbpo_main_config),
         deepcopy(pendulum_stevesac_mbpo_create_config)]
    ]
    try:
        for config in configs:
            config[0].world_model.model.max_epochs_since_update = 0
            serial_pipeline_dream(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

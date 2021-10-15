import pytest
import time
import os
from copy import deepcopy

from ding.entry import serial_pipeline_onpolicy
from dizoo.smac.config.smac_3s5z_mappo_config import main_config, create_config  # noqa


@pytest.mark.unittest
def test_mappo():
    config = [deepcopy(main_config), deepcopy(create_config)]
    config[0].policy.learn.epoch_per_collect = 1
    try:
        serial_pipeline_onpolicy(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"

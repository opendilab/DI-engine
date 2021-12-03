import pytest
from easydict import EasyDict
from copy import deepcopy
from ding.entry import serial_pipeline
from dizoo.mujoco.config.hopper_csql_default_config import hopper_csql_default_config, hopper_csql_default_create_config


@pytest.mark.unittest
def test_continous_sql():
    config = [deepcopy(hopper_csql_default_config), deepcopy(hopper_csql_default_create_config)]
    try:
        serial_pipeline(config, seed=0, max_iterations=2)
    except Exception:
        assert False, "pipeline fail"


if __name__ == "__main__":
    test_continous_sql()

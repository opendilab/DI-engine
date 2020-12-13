import pytest
import time
import os
from nervex.entry import serial_pipeline
from nervex.utils import read_config


@pytest.mark.unittest
def test_dqn():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_dqn_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ddpg():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/pendulum/entry/pendulum_ddpg_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

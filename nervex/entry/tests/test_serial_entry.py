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


def test_pong_dqn():
    path = os.path.join(os.path.dirname(__file__), '../../../app_zoo/atari/entry/pong_dqn_default_config.yaml')
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


@pytest.mark.unittest
def test_td3():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/pendulum/entry/pendulum_td3_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_a2c():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_a2c_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_rainbow_dqn():
    path = os.path.join(
        os.path.dirname(__file__),
        '../../../app_zoo/classic_control/cartpole/entry/cartpole_rainbowdqn_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_dqn_vanilla():
    path = os.path.join(
        os.path.dirname(__file__),
        '../../../app_zoo/classic_control/cartpole/entry/cartpole_dqnvanilla_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_ppo_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


def test_sac():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/pendulum/entry/pendulum_sac_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


# @pytest.mark.unittest
def test_r2d2():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_r2d2_default_config.yaml'
    )
    config = read_config(path)
    #config.evaluator.stop_val = 40  # for save time
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

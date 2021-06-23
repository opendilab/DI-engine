import pytest
import time
import os
from copy import deepcopy

from nervex.entry import serial_pipeline
from app_zoo.classic_control.lunarlander.tests import lunarlander_qrdqn_config, lunarlander_qrdqn_create_config


@pytest.mark.unittest
def test_qrdqn():
    config = [deepcopy(lunarlander_qrdqn_config), deepcopy(lunarlander_qrdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


if __name__ == "__main__":
    test_qrdqn()

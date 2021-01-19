import time
import os
from nervex.entry import serial_pipeline
from nervex.utils import read_config


def test_her():
    path = os.path.join(
        os.path.dirname(__file__), 'bitflip_dqnvanilla_default_config.yaml'
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=123)
    except Exception:
        assert False, "pipeline fail"




if __name__ == '__main__':
    test_her()


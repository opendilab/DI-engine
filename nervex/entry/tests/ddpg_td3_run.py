import time
import os
import argparse

from nervex.entry import serial_pipeline
from nervex.utils import read_config
from app_zoo.mujoco.model.q_ac_td3paper import QAC_td3paper


def test(prefix):
    path = os.path.join(
        os.path.dirname(__file__),
        '../../../../app_zoo/mujoco/entry/mujoco_single_machine/{}_default_config.yaml'.format(prefix)
    )
    config = read_config(path)
    try:
        serial_pipeline(config, seed=0, model_type=QAC_td3paper)
    except Exception:
        assert False, "pipeline fail"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ddpg')
    # ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']  # stop_val: 10500, 3700, 5300
    parser.add_argument('--env', type=str, default='halfcheetah')
    args = parser.parse_args()
    prefix = args.env + '_' + args.algo
    print('========={}======='.format(prefix))
    test(prefix)

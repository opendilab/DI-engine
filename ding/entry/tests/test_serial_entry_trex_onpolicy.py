import pytest
from copy import deepcopy
import os
from easydict import EasyDict

import torch

from ding.entry import serial_pipeline_onpolicy
from ding.entry.serial_entry_trex_onpolicy import serial_pipeline_reward_model_trex_onpolicy
from dizoo.mujoco.config import hopper_ppo_default_config, hopper_ppo_create_default_config
from dizoo.mujoco.config import hopper_trex_ppo_default_config, hopper_trex_ppo_create_default_config
from ding.entry.application_entry_trex_collect_data import trex_collecting_data

@pytest.mark.unittest
def test_serial_pipeline_reward_model_trex():
    config = [deepcopy(hopper_ppo_default_config), deepcopy(hopper_ppo_create_default_config)]
    expert_policy = serial_pipeline_onpolicy(config, seed=0, max_iterations=90)

    config = [deepcopy(hopper_trex_ppo_default_config), deepcopy(hopper_trex_ppo_create_default_config)]
    config[0].reward_model.offline_data_path = 'dizoo/mujoco/config/hopper_trex_onppo'
    config[0].reward_model.offline_data_path = os.path.abspath(config[0].reward_model.offline_data_path)
    config[0].reward_model.reward_model_path = config[0].reward_model.offline_data_path + '/hopper.params'
    config[0].reward_model.expert_model_path = './hopper_onppo'
    config[0].reward_model.expert_model_path = os.path.abspath(config[0].reward_model.expert_model_path)
    args = EasyDict(
        {
            'cfg': deepcopy(config),
            'seed': 0,
            'device': 'cpu'
        }
    )
    trex_collecting_data(args=args)
    try:
        serial_pipeline_reward_model_trex_onpolicy(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"

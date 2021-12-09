import pytest
from copy import deepcopy
import os
from easydict import EasyDict

import torch

from ding.entry import serial_pipeline
from ding.entry.serial_entry_trex import serial_pipeline_reward_model_trex
from dizoo.classic_control.cartpole.config.cartpole_trex_offppo_config import cartpole_trex_ppo_offpolicy_config,\
     cartpole_trex_ppo_offpolicy_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config,\
     cartpole_ppo_offpolicy_create_config
from ding.entry.application_entry_trex_collect_data import trex_collecting_data


# @pytest.mark.unittest
def test_serial_pipeline_reward_model_trex():
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    expert_policy = serial_pipeline(config, seed=0)

    config = [deepcopy(cartpole_trex_ppo_offpolicy_config), deepcopy(cartpole_trex_ppo_offpolicy_create_config)]
    config[0].reward_model.offline_data_path = 'dizoo/classic_control/cartpole/config/cartpole_trex_offppo'
    config[0].reward_model.offline_data_path = os.path.abspath(config[0].reward_model.offline_data_path)
    config[0].reward_model.reward_model_path = config[0].reward_model.offline_data_path + '/cartpole.params'
    config[0].reward_model.expert_model_path = './cartpole_ppo_offpolicy'
    config[0].reward_model.expert_model_path = os.path.abspath(config[0].reward_model.expert_model_path)
    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    trex_collecting_data(args=args)
    try:
        serial_pipeline_reward_model_trex(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"

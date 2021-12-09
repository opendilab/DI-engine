import pytest
from copy import deepcopy
import os
from easydict import EasyDict

import torch

from ding.entry import serial_pipeline_onpolicy
from ding.entry.serial_entry_trex_onpolicy import serial_pipeline_reward_model_trex_onpolicy
from dizoo.classic_control.cartpole.config import cartpole_ppo_config, cartpole_ppo_create_config
from dizoo.classic_control.cartpole.config import cartpole_trex_ppo_onpolicy_config, \
    cartpole_trex_ppo_onpolicy_create_config
from ding.entry.application_entry_trex_collect_data import trex_collecting_data


@pytest.mark.unittest
def test_serial_pipeline_reward_model_trex():
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    expert_policy = serial_pipeline_onpolicy(config, seed=0, max_iterations=90)

    config = [deepcopy(cartpole_trex_ppo_onpolicy_config), deepcopy(cartpole_trex_ppo_onpolicy_create_config)]
    config[0].reward_model.offline_data_path = 'dizoo/classic_control/cartpole/config/cartpole_trex_onppo'
    config[0].reward_model.offline_data_path = os.path.abspath(config[0].reward_model.offline_data_path)
    config[0].reward_model.reward_model_path = config[0].reward_model.offline_data_path + '/cartpole.params'
    config[0].reward_model.expert_model_path = './cartpole_ppo'
    config[0].reward_model.expert_model_path = os.path.abspath(config[0].reward_model.expert_model_path)
    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    trex_collecting_data(args=args)
    try:
        serial_pipeline_reward_model_trex_onpolicy(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"

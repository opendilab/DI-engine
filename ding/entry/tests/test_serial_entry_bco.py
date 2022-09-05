import pytest
import torch
from copy import deepcopy
from ding.entry import serial_pipeline
from ding.entry.serial_entry_bco import serial_pipeline_bco
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_bco_config import cartpole_bco_config, cartpole_bco_create_config


@pytest.mark.unittest
def test_bco():
    expert_policy_state_dict_path = './expert_policy.pth'
    expert_config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    expert_policy = serial_pipeline(expert_config, seed=0)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    config = [deepcopy(cartpole_bco_config), deepcopy(cartpole_bco_create_config)]
    config[0].policy.collect.model_path = expert_policy_state_dict_path
    try:
        serial_pipeline_bco(
            config, [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)], seed=0, max_train_iter=3
        )
    except Exception as e:
        print(e)
        assert False, "pipeline fail"

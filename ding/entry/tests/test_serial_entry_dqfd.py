import pytest
import torch
from copy import deepcopy
from ding.entry import serial_pipeline
from ding.entry.serial_entry_dqfd import serial_pipeline_dqfd
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_dqfd_config import cartpole_dqfd_config, cartpole_dqfd_create_config


@pytest.mark.unittest
def test_dqfd():
    expert_policy_state_dict_path = './expert_policy.pth'
    config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    expert_policy = serial_pipeline(config, seed=0)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    config = [deepcopy(cartpole_dqfd_config), deepcopy(cartpole_dqfd_create_config)]
    config[0].policy.collect.model_path = expert_policy_state_dict_path
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline_dqfd(config, [cartpole_dqfd_config, cartpole_dqfd_create_config], seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

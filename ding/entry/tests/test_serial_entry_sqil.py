import pytest
import torch
from copy import deepcopy
from ding.entry import serial_pipeline
from ding.entry.serial_entry_sqil import serial_pipeline_sqil
from dizoo.classic_control.cartpole.config.cartpole_sql_config import cartpole_sql_config, cartpole_sql_create_config
from dizoo.classic_control.cartpole.config.cartpole_sqil_config import cartpole_sqil_config, cartpole_sqil_create_config


@pytest.mark.unittest
def test_sqil():
    expert_policy_state_dict_path = './expert_policy.pth'
    config = [deepcopy(cartpole_sql_config), deepcopy(cartpole_sql_create_config)]
    expert_policy = serial_pipeline(config, seed=0)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    config = [deepcopy(cartpole_sqil_config), deepcopy(cartpole_sqil_create_config)]
    config[0].policy.collect.model_path = expert_policy_state_dict_path
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline_sqil(config, [cartpole_sql_config, cartpole_sql_create_config], seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

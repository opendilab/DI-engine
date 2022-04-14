import pytest
import torch
from copy import deepcopy
from ding.entry import serial_pipeline_onpolicy, serial_pipeline_guided_cost
from dizoo.classic_control.cartpole.config import cartpole_ppo_config, cartpole_ppo_create_config
from dizoo.classic_control.cartpole.config import cartpole_gcl_ppo_onpolicy_config, \
    cartpole_gcl_ppo_onpolicy_create_config


@pytest.mark.unittest
def test_guided_cost():
    expert_policy_state_dict_path = './expert_policy.pth'
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    expert_policy = serial_pipeline_onpolicy(config, seed=0)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    config = [deepcopy(cartpole_gcl_ppo_onpolicy_config), deepcopy(cartpole_gcl_ppo_onpolicy_create_config)]
    config[0].policy.collect.model_path = expert_policy_state_dict_path
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline_guided_cost(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

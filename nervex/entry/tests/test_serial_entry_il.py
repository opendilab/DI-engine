from copy import deepcopy

import pytest

from app_zoo.classic_control.cartpole.entry import cartpole_dqn_default_config, cartpole_a2c_default_config
from nervex.entry import serial_pipeline_il, collect_demo_data, serial_pipeline


@pytest.mark.unittest
def test_serial_pipeline_il():
    # train expert policy
    config = deepcopy(cartpole_a2c_default_config)
    expert_policy = serial_pipeline(config, seed=0)
    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    config = deepcopy(cartpole_a2c_default_config)
    collect_demo_data(
        config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )
    # il training
    config = deepcopy(cartpole_dqn_default_config)
    config.policy.learn.train_epoch = 10
    serial_pipeline_il(config, seed=0, data_path=expert_data_path)

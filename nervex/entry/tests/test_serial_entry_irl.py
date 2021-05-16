from copy import deepcopy
import pytest
import os
import logging

from app_zoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from app_zoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config, cartpole_ppo_create_config
from nervex.entry import serial_pipeline, collect_demo_data, serial_pipeline_irl

cfg = [
    {
        'type': 'pdeil',
        "alpha": 0.5,
        "discrete_action": False
    },
    {
        'type': 'gail',
        'input_dims': 5,
        'hidden_dims': 64,
        'batch_size': 64,
        'update_per_collect': 100
    },
]


@pytest.mark.unittest
@pytest.mark.parametrize('irl_config', cfg)
def test_pdeil(irl_config):
    config = deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)
    expert_policy = serial_pipeline(config, seed=0)
    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    config = deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)
    collect_demo_data(
        config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )
    # irl + rl training
    cp_cartpole_dqn_config = deepcopy(cartpole_dqn_config)
    cp_cartpole_dqn_create_config = deepcopy(cartpole_dqn_create_config)
    cp_cartpole_dqn_config.policy.learn.init_data_count = 10000
    irl_config['expert_data_path'] = expert_data_path
    cp_cartpole_dqn_config.irl = irl_config
    if irl_config['type'] == 'gail':
        cp_cartpole_dqn_config.policy.collect.n_sample = irl_config['batch_size']
    serial_pipeline_irl((cp_cartpole_dqn_config, cp_cartpole_dqn_create_config), seed=0)

    os.popen("rm -rf ckpt_* log expert_data.pkl")


if __name__ == '__main__':
    pytest.main(["-sv", "test_serial_entry_irl.py"])

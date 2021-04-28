from copy import deepcopy

import pytest

from app_zoo.classic_control.cartpole.entry import cartpole_ppo_default_config, cartpole_dqn_default_config
from nervex.entry import serial_pipeline_irl, collect_demo_data, serial_pipeline

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
        'train_iterations': 100
    },
]


@pytest.mark.unittest
@pytest.mark.parametrize('irl_config', cfg)
def test_irl(irl_config):
    # train expert policy
    config = deepcopy(cartpole_ppo_default_config)
    expert_policy = serial_pipeline(config, seed=0)
    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    config = deepcopy(cartpole_ppo_default_config)
    collect_demo_data(
        config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )
    # irl + rl training
    config = deepcopy(cartpole_dqn_default_config)
    config.policy.learn.init_data_count = 10000
    irl_config['expert_data_path'] = expert_data_path
    config.irl = irl_config
    if irl_config['type'] == 'gail':
        config.collector.n_sample = irl_config['batch_size']
    serial_pipeline_irl(config, seed=0)


if __name__ == '__main__':
    pytest.main(["-sv", "test_serial_entry_irl.py"])

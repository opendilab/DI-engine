import pytest
from copy import deepcopy
from app_zoo.classic_control.cartpole.entry import cartpole_ppo_default_config, cartpole_dqn_default_config
from nervex.entry import serial_pipeline_irl, collect_demo_data, serial_pipeline


@pytest.mark.unittest
def test_pdeil():
    # train expert policy
    config = deepcopy(cartpole_ppo_default_config)
    expert_policy = serial_pipeline(config, seed=0)
    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = {'model': expert_policy.state_dict_handle()['model'].state_dict()}
    config = deepcopy(cartpole_ppo_default_config)
    collect_demo_data(
        config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )
    # irl + rl training
    config = deepcopy(cartpole_dqn_default_config)
    config.policy.learn.init_data_count = 10000
    config.irl = {"alpha": 0.5, "expert_data_path": expert_data_path, "discrete_action": False}
    serial_pipeline_irl(config, seed=0)

from copy import deepcopy
import pytest
import os
import pickle

from dizoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config, cartpole_ppo_create_config
from ding.entry import serial_pipeline, eval, collect_demo_data
from ding.config import compile_config


@pytest.fixture(scope='module')
def setup_state_dict():
    config = deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)
    try:
        policy = serial_pipeline(config, seed=0)
    except Exception:
        assert False, 'Serial pipeline failure'
    state_dict = {
        'eval': policy.eval_mode.state_dict(),
        'collect': policy.collect_mode.state_dict(),
    }
    return state_dict


@pytest.mark.unittest
class TestApplication:

    def test_eval(self, setup_state_dict):
        cfg_for_stop_value = compile_config(cartpole_ppo_config, auto=True, create_cfg=cartpole_ppo_create_config)
        stop_value = cfg_for_stop_value.env.stop_value
        config = deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)
        eval_reward = eval(config, seed=0, state_dict=setup_state_dict['eval'])
        assert eval_reward >= stop_value

    def test_collect_demo_data(self, setup_state_dict):
        config = deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)
        collect_count = 16
        expert_data_path = './expert.data'
        collect_demo_data(
            config,
            seed=0,
            state_dict=setup_state_dict['collect'],
            collect_count=collect_count,
            expert_data_path=expert_data_path
        )
        with open(expert_data_path, 'rb') as f:
            exp_data = pickle.load(f)
        assert isinstance(exp_data, list)
        assert isinstance(exp_data[0], dict)
        os.popen('rm -rf ./expert.data ckpt* log')

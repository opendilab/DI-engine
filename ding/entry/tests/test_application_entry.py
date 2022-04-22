from copy import deepcopy
import pytest
import os
import pickle

from dizoo.classic_control.cartpole.config.cartpole_offppo_config import cartpole_offppo_config, \
    cartpole_offppo_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_trex_offppo_config import cartpole_trex_offppo_config,\
     cartpole_trex_offppo_create_config
from dizoo.classic_control.cartpole.envs import CartPoleEnv
from ding.entry import serial_pipeline, eval, collect_demo_data
from ding.config import compile_config
from ding.entry.application_entry import collect_episodic_demo_data, episode_to_transitions


@pytest.fixture(scope='module')
def setup_state_dict():
    config = deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)
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
        cfg_for_stop_value = compile_config(cartpole_offppo_config, auto=True, create_cfg=cartpole_offppo_create_config)
        stop_value = cfg_for_stop_value.env.stop_value
        config = deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)
        eval_reward = eval(config, seed=0, state_dict=setup_state_dict['eval'])
        assert eval_reward >= stop_value
        config = deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)
        eval_reward = eval(
            config,
            seed=0,
            env_setting=[CartPoleEnv, None, [{} for _ in range(5)]],
            state_dict=setup_state_dict['eval']
        )
        assert eval_reward >= stop_value

    def test_collect_demo_data(self, setup_state_dict):
        config = deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)
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

    def test_collect_episodic_demo_data(self, setup_state_dict):
        config = deepcopy(cartpole_trex_offppo_config), deepcopy(cartpole_trex_offppo_create_config)
        config[0].exp_name = 'cartpole_trex_offppo_episodic'
        collect_count = 16
        if not os.path.exists('./test_episode'):
            os.mkdir('./test_episode')
        expert_data_path = './test_episode/expert.data'
        collect_episodic_demo_data(
            config,
            seed=0,
            state_dict=setup_state_dict['collect'],
            expert_data_path=expert_data_path,
            collect_count=collect_count,
        )
        with open(expert_data_path, 'rb') as f:
            exp_data = pickle.load(f)
        assert isinstance(exp_data, list)
        assert isinstance(exp_data[0][0], dict)

    def test_episode_to_transitions(self, setup_state_dict):
        self.test_collect_episodic_demo_data(setup_state_dict)
        expert_data_path = './test_episode/expert.data'
        episode_to_transitions(data_path=expert_data_path, expert_data_path=expert_data_path, nstep=3)
        with open(expert_data_path, 'rb') as f:
            exp_data = pickle.load(f)
        assert isinstance(exp_data, list)
        assert isinstance(exp_data[0], dict)
        os.popen('rm -rf ./test_episode/expert.data ckpt* log')
        os.popen('rm -rf ./test_episode')

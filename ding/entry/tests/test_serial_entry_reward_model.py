import pytest
import os
from ditk import logging
from easydict import EasyDict
from copy import deepcopy

from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_offppo_config import cartpole_offppo_config, cartpole_offppo_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_rnd_onppo_config import cartpole_ppo_rnd_config, cartpole_ppo_rnd_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_ppo_icm_config import cartpole_ppo_icm_config, cartpole_ppo_icm_create_config  # noqa
from ding.entry import serial_pipeline, collect_demo_data, serial_pipeline_reward_model_offpolicy, \
    serial_pipeline_reward_model_onpolicy

cfg = [
    {
        'type': 'pdeil',
        "alpha": 0.5,
        "discrete_action": False
    },
    {
        'type': 'gail',
        'input_size': 5,
        'hidden_size': 64,
        'batch_size': 64,
    },
    {
        'type': 'pwil',
        's_size': 4,
        'a_size': 2,
        'sample_size': 500,
    },
    {
        'type': 'red',
        'sample_size': 5000,
        'input_size': 5,
        'hidden_size': 64,
        'update_per_collect': 200,
        'batch_size': 128,
    },
]


@pytest.mark.unittest
@pytest.mark.parametrize('reward_model_config', cfg)
def test_irl(reward_model_config):
    reward_model_config = EasyDict(reward_model_config)
    config = deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)
    expert_policy = serial_pipeline(config, seed=0, max_train_iter=2)
    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    config = deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)
    collect_demo_data(
        config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )
    # irl + rl training
    cp_cartpole_dqn_config = deepcopy(cartpole_dqn_config)
    cp_cartpole_dqn_create_config = deepcopy(cartpole_dqn_create_config)
    cp_cartpole_dqn_create_config.reward_model = dict(type=reward_model_config.type)
    if reward_model_config.type == 'gail':
        reward_model_config['data_path'] = '.'
    else:
        reward_model_config['expert_data_path'] = expert_data_path
    cp_cartpole_dqn_config.reward_model = reward_model_config
    cp_cartpole_dqn_config.policy.collect.n_sample = 128
    serial_pipeline_reward_model_offpolicy(
        (cp_cartpole_dqn_config, cp_cartpole_dqn_create_config), seed=0, max_train_iter=2
    )

    os.popen("rm -rf ckpt_* log expert_data.pkl")


@pytest.mark.unittest
def test_rnd():
    config = [deepcopy(cartpole_ppo_rnd_config), deepcopy(cartpole_ppo_rnd_create_config)]
    try:
        serial_pipeline_reward_model_onpolicy(config, seed=0, max_train_iter=2)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_icm():
    config = [deepcopy(cartpole_ppo_icm_config), deepcopy(cartpole_ppo_icm_create_config)]
    try:
        serial_pipeline_reward_model_offpolicy(config, seed=0, max_train_iter=2)
    except Exception:
        assert False, "pipeline fail"

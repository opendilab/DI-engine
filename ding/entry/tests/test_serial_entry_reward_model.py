import pytest
import os
from ditk import logging
from easydict import EasyDict
from copy import deepcopy

from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_trex_offppo_config import cartpole_trex_offppo_config,\
     cartpole_trex_offppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config, cartpole_ppo_offpolicy_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_rnd_onppo_config import cartpole_ppo_rnd_config, cartpole_ppo_rnd_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_ppo_icm_config import cartpole_ppo_icm_config, cartpole_ppo_icm_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_ngu_config import cartpole_ngu_config, cartpole_ngu_create_config
from ding.entry import serial_pipeline, collect_demo_data, serial_pipeline_reward_model_offpolicy, \
    serial_pipeline_reward_model_onpolicy
from ding.entry.application_entry_trex_collect_data import trex_collecting_data

cfg = [
    {
        'type': 'pdeil',
        "alpha": 0.5,
        "discrete_action": False
    }, {
        'type': 'gail',
        'input_size': 5,
        'hidden_size_list': [64],
        'batch_size': 64,
    }, {
        'type': 'pwil',
        's_size': 4,
        'a_size': 2,
        'sample_size': 500,
    }, {
        'type': 'red',
        'sample_size': 5000,
        'obs_shape': 4,
        'action_shape': 1,
        'hidden_size_list': [64, 1],
        'update_per_collect': 200,
        'batch_size': 128,
    }
]


@pytest.mark.unittest
@pytest.mark.parametrize('reward_model_config', cfg)
def test_irl(reward_model_config):
    reward_model_config = EasyDict(reward_model_config)
    config = deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)
    expert_policy = serial_pipeline(config, seed=0, max_train_iter=2)
    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    config = deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)
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
    cooptrain_reward = True
    pretrain_reward = False
    serial_pipeline_reward_model_offpolicy(
        (cp_cartpole_dqn_config, cp_cartpole_dqn_create_config),
        seed=0,
        max_train_iter=2,
        pretrain_reward=pretrain_reward,
        cooptrain_reward=cooptrain_reward
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


@pytest.mark.unittest
def test_ngu():
    config = [deepcopy(cartpole_ngu_config), deepcopy(cartpole_ngu_create_config)]
    try:
        serial_pipeline_reward_model_offpolicy(config, seed=0, max_train_iter=2)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_trex():
    exp_name = 'test_serial_pipeline_trex_expert'
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    config[0].exp_name = exp_name
    expert_policy = serial_pipeline(config, seed=0)

    exp_name = 'test_serial_pipeline_trex_collect'
    config = [deepcopy(cartpole_trex_offppo_config), deepcopy(cartpole_trex_offppo_create_config)]
    config[0].exp_name = exp_name
    config[0].reward_model.exp_name = exp_name
    config[0].reward_model.expert_model_path = 'test_serial_pipeline_trex_expert'
    config[0].reward_model.checkpoint_max = 100
    config[0].reward_model.checkpoint_step = 100
    config[0].reward_model.num_snippets = 100
    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    trex_collecting_data(args=args)
    try:
        serial_pipeline_reward_model_offpolicy(
            config, seed=0, max_train_iter=1, pretrain_reward=True, cooptrain_reward=False
        )
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf test_serial_pipeline_trex*')

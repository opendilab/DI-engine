from easydict import EasyDict
import pytest
from copy import deepcopy
import os
from itertools import product

import torch

from dizoo.classic_control.cartpole.config.cartpole_trex_offppo_config import cartpole_trex_offppo_config,\
     cartpole_trex_offppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_offppo_config import cartpole_offppo_config,\
     cartpole_offppo_create_config
from ding.entry.application_entry_trex_collect_data import collect_episodic_demo_data_for_trex, trex_collecting_data
from ding.entry import serial_pipeline


@pytest.mark.unittest
def test_collect_episodic_demo_data_for_trex():
    expert_policy_state_dict_path = './expert_policy.pth'
    expert_policy_state_dict_path = os.path.abspath('./expert_policy.pth')
    config = [deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)]
    expert_policy = serial_pipeline(config, seed=0)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    config = deepcopy(cartpole_trex_offppo_config), deepcopy(cartpole_trex_offppo_create_config)
    collect_count = 1
    save_cfg_path = './cartpole_trex_offppo'
    save_cfg_path = os.path.abspath(save_cfg_path)
    exp_data = collect_episodic_demo_data_for_trex(
        config,
        seed=0,
        state_dict_path=expert_policy_state_dict_path,
        save_cfg_path=save_cfg_path,
        collect_count=collect_count,
        rank=1,
    )
    assert isinstance(exp_data, list)
    assert isinstance(exp_data[0][0], dict)
    os.popen('rm -rf {}'.format(save_cfg_path))
    os.popen('rm -rf {}'.format(expert_policy_state_dict_path))


@pytest.mark.unittest
def test_trex_collecting_data():
    expert_policy_state_dict_path = './cartpole_offppo_seed0'
    expert_policy_state_dict_path = os.path.abspath(expert_policy_state_dict_path)
    config = [deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)]
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    expert_policy = serial_pipeline(config, seed=0)

    args = EasyDict(
        {
            'cfg': [deepcopy(cartpole_trex_offppo_config),
                    deepcopy(cartpole_trex_offppo_create_config)],
            'seed': 0,
            'device': 'cpu'
        }
    )
    args.cfg[0].reward_model.data_path = './cartpole_trex_offppo_seed0'
    args.cfg[0].reward_model.data_path = os.path.abspath(args.cfg[0].reward_model.data_path)
    args.cfg[0].reward_model.reward_model_path = args.cfg[0].reward_model.data_path + '/cartpole.params'
    args.cfg[0].reward_model.expert_model_path = './cartpole_offppo_seed0'
    args.cfg[0].reward_model.expert_model_path = os.path.abspath(args.cfg[0].reward_model.expert_model_path)
    args.cfg[0].reward_model.checkpoint_max = 100
    args.cfg[0].reward_model.checkpoint_step = 100
    args.cfg[0].reward_model.num_snippets = 100
    trex_collecting_data(args=args)
    os.popen('rm -rf {}'.format(expert_policy_state_dict_path))
    os.popen('rm -rf {}'.format(args.cfg[0].reward_model.data_path))

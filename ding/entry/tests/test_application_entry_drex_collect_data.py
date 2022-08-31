from easydict import EasyDict
from copy import deepcopy
from itertools import product

import pytest
import os
import torch

from ding.entry import serial_pipeline
from ding.entry.application_entry_drex_collect_data import collect_episodic_demo_data_for_drex, drex_collecting_data
from ding.config import compile_config
from dizoo.classic_control.cartpole.config.cartpole_drex_dqn_config import cartpole_drex_dqn_config,\
     cartpole_drex_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config,\
     cartpole_dqn_create_config


@pytest.mark.unittest
def test_collect_episodic_demo_data_for_drex():
    expert_policy_state_dict_path = './expert_policy.pth'
    expert_policy_state_dict_path = os.path.abspath('./expert_policy.pth')
    config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    expert_policy = serial_pipeline(config, seed=0, max_train_iter=1)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    config = deepcopy(cartpole_drex_dqn_config)
    config = compile_config(
        config, seed=0, env=None, auto=True, create_cfg=cartpole_drex_dqn_create_config, save_cfg=True
    )
    collect_count = 1
    save_cfg_path = './cartpole_drex_dqn'
    save_cfg_path = os.path.abspath(save_cfg_path)
    exp_data = collect_episodic_demo_data_for_drex(
        config,
        seed=0,
        state_dict_path=expert_policy_state_dict_path,
        save_cfg_path=save_cfg_path,
        collect_count=collect_count,
        rank=1,
        noise=-1,
    )
    assert isinstance(exp_data, list)
    assert isinstance(exp_data[0][0], dict)
    os.popen('rm -rf {}'.format(save_cfg_path))
    os.popen('rm -rf {}'.format(expert_policy_state_dict_path))


@pytest.mark.unittest
def test_drex_collecting_data():
    expert_policy_state_dict_path = './cartpole_dqn_seed0'
    expert_policy_state_dict_path = os.path.abspath(expert_policy_state_dict_path)
    config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    expert_policy = serial_pipeline(config, seed=0, max_train_iter=1)

    args = EasyDict(
        {
            'cfg': [deepcopy(cartpole_drex_dqn_config),
                    deepcopy(cartpole_drex_dqn_create_config)],
            'seed': 0,
            'device': 'cpu'
        }
    )
    args.cfg[0].reward_model.offline_data_path = './cartpole_drex_dqn_seed0'
    args.cfg[0].reward_model.offline_data_path = os.path.abspath(args.cfg[0].reward_model.offline_data_path)
    args.cfg[0].reward_model.reward_model_path = args.cfg[0].reward_model.offline_data_path + '/cartpole.params'
    args.cfg[0].reward_model.expert_model_path = './cartpole_dqn_seed0/ckpt/ckpt_best.pth.tar'
    args.cfg[0].reward_model.expert_model_path = os.path.abspath(args.cfg[0].reward_model.expert_model_path)
    args.cfg[0].reward_model.bc_iterations = 6
    args.cfg[0].reward_model.num_trajs_per_bin = 8
    args.cfg[0].bc_iteration = 1000  # for unittest
    args.cfg[1].policy.type = 'bc'
    drex_collecting_data(args=args)
    os.popen('rm -rf {}'.format(expert_policy_state_dict_path))
    os.popen('rm -rf {}'.format(args.cfg[0].reward_model.offline_data_path))

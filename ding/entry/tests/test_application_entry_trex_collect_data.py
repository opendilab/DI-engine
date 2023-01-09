from easydict import EasyDict
import pytest
from copy import deepcopy
import os
from itertools import product

import torch

from dizoo.classic_control.cartpole.config.cartpole_trex_offppo_config import cartpole_trex_offppo_config,\
     cartpole_trex_offppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config,\
     cartpole_ppo_offpolicy_create_config
from ding.entry.application_entry_trex_collect_data import collect_episodic_demo_data_for_trex, trex_collecting_data
from ding.entry import serial_pipeline


@pytest.mark.unittest
def test_collect_episodic_demo_data_for_trex():
    exp_name = "test_collect_episodic_demo_data_for_trex_expert"
    expert_policy_state_dict_path = os.path.join(exp_name, 'expert_policy.pth.tar')
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    config[0].exp_name = exp_name
    expert_policy = serial_pipeline(config, seed=0)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    exp_name = "test_collect_episodic_demo_data_for_trex_collect"
    config = [deepcopy(cartpole_trex_offppo_config), deepcopy(cartpole_trex_offppo_create_config)]
    config[0].exp_name = exp_name
    exp_data = collect_episodic_demo_data_for_trex(
        config,
        seed=0,
        state_dict_path=expert_policy_state_dict_path,
        collect_count=1,
        rank=1,
    )
    assert isinstance(exp_data, list)
    assert isinstance(exp_data[0][0], dict)
    os.popen('rm -rf test_collect_episodic_demo_data_for_trex*')


@pytest.mark.unittest
def test_trex_collecting_data():
    expert_policy_dir = 'test_trex_collecting_data_expert'
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    config[0].exp_name = expert_policy_dir
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    serial_pipeline(config, seed=0)

    args = EasyDict(
        {
            'cfg': [deepcopy(cartpole_trex_offppo_config),
                    deepcopy(cartpole_trex_offppo_create_config)],
            'seed': 0,
            'device': 'cpu'
        }
    )
    exp_name = 'test_trex_collecting_data_collect'
    args.cfg[0].exp_name = exp_name
    args.cfg[0].reward_model.reward_model_path = os.path.join(exp_name, "reward_model.pth.tar")
    args.cfg[0].reward_model.expert_model_path = expert_policy_dir
    args.cfg[0].reward_model.checkpoint_max = 100
    args.cfg[0].reward_model.checkpoint_step = 100
    args.cfg[0].reward_model.num_snippets = 100
    trex_collecting_data(args=args)
    os.popen('rm -rf test_trex_collecting_data*')

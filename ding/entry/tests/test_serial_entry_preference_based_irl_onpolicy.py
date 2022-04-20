import pytest
from copy import deepcopy
import os
from easydict import EasyDict

import torch

from ding.entry import serial_pipeline_onpolicy
from ding.entry import serial_pipeline_preference_based_irl_onpolicy
from dizoo.classic_control.cartpole.config import cartpole_ppo_config, cartpole_ppo_create_config
from dizoo.classic_control.cartpole.config import cartpole_trex_ppo_onpolicy_config, \
    cartpole_trex_ppo_onpolicy_create_config
from ding.entry.application_entry_trex_collect_data import trex_collecting_data


@pytest.mark.unittest
def test_serial_pipeline_reward_model_trex():
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    expert_policy = serial_pipeline_onpolicy(config, seed=0)

    config = [deepcopy(cartpole_trex_ppo_onpolicy_config), deepcopy(cartpole_trex_ppo_onpolicy_create_config)]
    config[0].reward_model.data_path = './cartpole_trex_onppo_seed0'
    config[0].reward_model.data_path = os.path.abspath(config[0].reward_model.data_path)
    config[0].reward_model.reward_model_path = config[0].reward_model.data_path + '/cartpole.params'
    config[0].reward_model.expert_model_path = './cartpole_onppo_seed0'
    config[0].reward_model.expert_model_path = os.path.abspath(config[0].reward_model.expert_model_path)
    config[0].reward_model.checkpoint_max = 100
    config[0].reward_model.checkpoint_step = 100
    config[0].reward_model.num_snippets = 100
    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    trex_collecting_data(args=args)
    try:
        serial_pipeline_preference_based_irl_onpolicy(config, seed=0, max_train_iter=1)
        os.popen('rm -rf {}'.format(config[0].reward_model.data_path))
        os.popen('rm -rf {}'.format(config[0].reward_model.expert_model_path))
    except Exception:
        assert False, "pipeline fail"

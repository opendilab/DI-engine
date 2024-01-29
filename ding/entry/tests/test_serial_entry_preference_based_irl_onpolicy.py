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
def test_serial_pipeline_trex_onpolicy():
    exp_name = 'trex_onpolicy_test_serial_pipeline_trex_onpolicy_expert'
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    config[0].exp_name = exp_name
    expert_policy = serial_pipeline_onpolicy(config, seed=0)

    exp_name = 'trex_onpolicy_test_serial_pipeline_trex_onpolicy_collect'
    config = [deepcopy(cartpole_trex_ppo_onpolicy_config), deepcopy(cartpole_trex_ppo_onpolicy_create_config)]
    config[0].exp_name = exp_name
    config[0].reward_model.expert_model_path = 'trex_onpolicy_test_serial_pipeline_trex_onpolicy_expert'
    config[0].reward_model.checkpoint_max = 100
    config[0].reward_model.checkpoint_step = 100
    config[0].reward_model.num_snippets = 100
    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    trex_collecting_data(args=args)
    try:
        serial_pipeline_preference_based_irl_onpolicy(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf test_serial_pipeline_trex_onpolicy*')

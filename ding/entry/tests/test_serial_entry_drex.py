import pytest
import os
from easydict import EasyDict
from copy import deepcopy

from dizoo.classic_control.cartpole.config.cartpole_dqn_config \
import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_drex_dqn_config \
import cartpole_drex_dqn_config, cartpole_drex_dqn_create_config
from ding.entry import serial_pipeline, serial_pipeline_reward_model_offpolicy
from ding.entry.application_entry_drex_collect_data import drex_collecting_data


@pytest.mark.unittest
def test_drex():
    exp_name = 'test_serial_pipeline_drex_expert'
    config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    config[0].exp_name = exp_name
    expert_policy = serial_pipeline(config, seed=0)

    exp_name = 'test_serial_pipeline_drex_collect'
    config = [deepcopy(cartpole_drex_dqn_config), deepcopy(cartpole_drex_dqn_create_config)]
    config[0].exp_name = exp_name
    config[0].reward_model.exp_name = exp_name
    config[0].reward_model.expert_model_path = 'test_serial_pipeline_drex_expert/ckpt/ckpt_best.pth.tar'
    config[0].reward_model.reward_model_path = 'test_serial_pipeline_drex_collect/cartpole.params'
    config[0].reward_model.offline_data_path = 'test_serial_pipeline_drex_collect'
    config[0].reward_model.checkpoint_max = 100
    config[0].reward_model.checkpoint_step = 100
    config[0].reward_model.num_snippets = 100

    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    args.cfg[0].policy.collect.n_episode = 8
    del args.cfg[0].policy.collect.n_sample
    args.cfg[0].bc_iteration = 1000  # for unittest
    args.cfg[1].policy.type = 'bc'
    drex_collecting_data(args=args)
    try:
        serial_pipeline_reward_model_offpolicy(
            config, seed=0, max_train_iter=1, pretrain_reward=True, cooptrain_reward=False
        )
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf test_serial_pipeline_drex*')

import pytest
from copy import deepcopy
import os
from easydict import EasyDict

import torch
import numpy as np

from ding.entry import serial_pipeline
from ding.entry.serial_entry_trex import serial_pipeline_reward_model_trex
from dizoo.classic_control.cartpole.config.cartpole_trex_offppo_config import cartpole_trex_ppo_offpolicy_config,\
     cartpole_trex_ppo_offpolicy_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config,\
     cartpole_ppo_offpolicy_create_config
from dizoo.atari.config.serial.pong.pong_ppo_offpolicy_config import pong_ppo_config, \
    pong_ppo_create_config
from dizoo.atari.config.serial.pong.pong_trex_offppo_config import pong_trex_ppo_config, \
    pong_trex_ppo_create_config
from ding.entry.application_entry_trex_collect_data import trex_collecting_data
from ding.reward_model.trex_reward_model import TrexConvEncoder
from ding.torch_utils import is_differentiable


@pytest.mark.unittest
def test_serial_pipeline_reward_model_trex():
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    expert_policy = serial_pipeline(config, seed=0)

    config = [deepcopy(cartpole_trex_ppo_offpolicy_config), deepcopy(cartpole_trex_ppo_offpolicy_create_config)]
    config[0].reward_model.offline_data_path = './cartpole_trex_offppo'
    config[0].reward_model.offline_data_path = os.path.abspath(config[0].reward_model.offline_data_path)
    config[0].reward_model.reward_model_path = config[0].reward_model.offline_data_path + '/cartpole.params'
    config[0].reward_model.expert_model_path = './cartpole_ppo_offpolicy'
    config[0].reward_model.expert_model_path = os.path.abspath(config[0].reward_model.expert_model_path)
    config[0].reward_model.checkpoint_max = 100
    config[0].reward_model.checkpoint_step = 100
    config[0].reward_model.num_snippets = 100
    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    trex_collecting_data(args=args)
    try:
        serial_pipeline_reward_model_trex(config, seed=0, max_iterations=1)
        os.popen('rm -rf {}'.format(config[0].reward_model.offline_data_path))
    except Exception:
        assert False, "pipeline fail"


B = 4
C, H, W = 3, 128, 128


@pytest.mark.unittest
class TestEncoder:

    def output_check(self, model, outputs):
        loss = outputs.sum()
        is_differentiable(loss, model)

    def test_conv_encoder(self):
        inputs = torch.randn(B, C, H, W)
        model = TrexConvEncoder((C, H, W))
        print(model)
        outputs = model(inputs)
        self.output_check(model, outputs)
        print(outputs.shape)
        assert outputs.shape == (B, 1)

import pytest
from copy import deepcopy
import os
from easydict import EasyDict

import torch

from ding.entry import serial_pipeline
from ding.entry import serial_pipeline_preference_based_irl
from dizoo.classic_control.cartpole.config.cartpole_trex_offppo_config import cartpole_trex_offppo_config,\
     cartpole_trex_offppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config,\
     cartpole_ppo_offpolicy_create_config
from ding.entry.application_entry_trex_collect_data import trex_collecting_data
from ding.reward_model.trex_reward_model import TrexConvEncoder
from ding.torch_utils import is_differentiable


@pytest.mark.unittest
def test_serial_pipeline_trex():
    exp_name = 'test_serial_pipeline_trex_expert'
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    config[0].policy.learn.learner.hook.save_ckpt_after_iter = 100
    config[0].exp_name = exp_name
    expert_policy = serial_pipeline(config, seed=0)

    exp_name = 'test_serial_pipeline_trex_collect'
    config = [deepcopy(cartpole_trex_offppo_config), deepcopy(cartpole_trex_offppo_create_config)]
    config[0].exp_name = exp_name
    config[0].reward_model.expert_model_path = 'test_serial_pipeline_trex_expert'
    config[0].reward_model.checkpoint_max = 100
    config[0].reward_model.checkpoint_step = 100
    config[0].reward_model.num_snippets = 100
    args = EasyDict({'cfg': deepcopy(config), 'seed': 0, 'device': 'cpu'})
    trex_collecting_data(args=args)
    try:
        serial_pipeline_preference_based_irl(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf test_serial_pipeline_trex*')


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

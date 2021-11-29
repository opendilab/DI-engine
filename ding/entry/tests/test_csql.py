import pytest
import os
import logging
from easydict import EasyDict
from copy import deepcopy

from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config, cartpole_ppo_offpolicy_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_ppo_rnd_config import cartpole_ppo_rnd_config, cartpole_ppo_rnd_create_config  # noqa
from ding.entry import serial_pipeline, collect_demo_data, serial_pipeline_reward_model


@pytest.mark.unittest
def test_rnd():
    config = [deepcopy(cartpole_ppo_rnd_config), deepcopy(cartpole_ppo_rnd_create_config)]
    try:
        serial_pipeline_reward_model(config, seed=0, max_iterations=2)
    except Exception:
        assert False, "pipeline fail"

if __name__ == "__main__":
    test_rnd()

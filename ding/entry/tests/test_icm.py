import pytest
from easydict import EasyDict
from copy import deepcopy
from ding.entry import serial_pipeline_reward_model
from dizoo.classic_control.cartpole.config.cartpole_ppo_icm_config import cartpole_ppo_icm_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_icm_config import cartpole_ppo_icm_create_config


@pytest.mark.unittest
def test_icm():
    config = [deepcopy(cartpole_ppo_icm_config), deepcopy(cartpole_ppo_icm_create_config)]
    try:
        serial_pipeline_reward_model(config, seed=0, max_iterations=2)
    except Exception:
        assert False, "pipeline fail"


if __name__ == '__main__':
    test_icm()

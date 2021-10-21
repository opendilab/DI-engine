import pytest
import time
import os
from copy import deepcopy

from ding.entry import serial_pipeline_onpolicy
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config, cartpole_ppo_create_config
from dizoo.multiagent_particle.config import cooperative_navigation_mappo_config, cooperative_navigation_mappo_create_config  # noqa


@pytest.mark.unittest
def test_on_ppo():
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    config[0].policy.learn.epoch_per_collect = 1
    try:
        serial_pipeline_onpolicy(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_on_ppo_recompute_adv():
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    config[0].policy.learn.epoch_per_collect = 1
    config[0].policy.recompute_adv = False
    try:
        serial_pipeline_onpolicy(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_mappo():
    config = [deepcopy(cooperative_navigation_mappo_config), deepcopy(cooperative_navigation_mappo_create_config)]
    config[0].policy.learn.epoch_per_collect = 1
    try:
        serial_pipeline_onpolicy(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"

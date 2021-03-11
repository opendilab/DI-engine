import pytest
from copy import deepcopy
from app_zoo.classic_control.cartpole.entry import cartpole_ppo_default_config, cartpole_dqn_default_config
from nervex.entry import serial_pipeline_irl


# @pytest.mark.unittest
def test_pdeil():
    config = deepcopy(cartpole_dqn_default_config)
    config.policy.learn.init_data_count = 10000
    config.irl = {
        "alpha": 0.5,
        "expert_data_path": '/Users/nyz/code/gitlab/nerveX/nervex/entry/tests/expert_data.pkl',
        "discrete_action": False
    }
    serial_pipeline_irl(config, seed=0)

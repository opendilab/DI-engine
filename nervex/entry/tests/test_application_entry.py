from copy import deepcopy

import pytest

from app_zoo.classic_control.cartpole.entry import cartpole_dqn_default_config
from nervex.entry import serial_pipeline, eval


@pytest.mark.unittest
def test_eval():
    config = deepcopy(cartpole_dqn_default_config)
    stop_value = config.evaluator.stop_value
    try:
        policy = serial_pipeline(config, seed=0)
    except Exception:
        assert False, 'Serial pipeline failure'
    state_dict = {'model': policy.state_dict_handle()['model'].state_dict()}
    eval_reward = eval(config, seed=0, state_dict=state_dict)
    assert eval_reward >= stop_value

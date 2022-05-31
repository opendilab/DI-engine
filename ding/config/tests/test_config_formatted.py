import pytest
import importlib
from typing import Union, Optional, List, Any, Callable, Tuple
from ding.config import read_config, compile_config
import dizoo.classic_control.cartpole.config.cartpole_ppo_config as cppo
import dizoo.classic_control.cartpole.config.cartpole_dqn_config as cdqn
import dizoo.classic_control.cartpole.config.cartpole_a2c_config as ca2c
import dizoo.classic_control.cartpole.config.cartpole_c51_config as cc51

args = [
    ['dizoo.classic_control.cartpole.config.cartpole_ppo_config', 'onppo'],
    ['dizoo.classic_control.cartpole.config.cartpole_a2c_config', 'a2c'],
    # TODO adapt to new buffer
    # ['dizoo.classic_control.cartpole.config.cartpole_dqn_config', 'dqn',
    ['dizoo.classic_control.cartpole.config.cartpole_c51_config', 'c51'],
]


@pytest.mark.unittest
@pytest.mark.parametrize('config_path, name', args)
def test_config_formatted(config_path, name):
    module_config = importlib.import_module(config_path)
    main_config, create_config = module_config.main_config, module_config.create_config
    cfg = compile_config(
        main_config, seed=0, auto=True, create_cfg=create_config, save_cfg=True, save_path='{}_config.py'.format(name)
    )

    module = importlib.import_module('cartpole_{}_seed0.formatted_{}_config'.format(name, name))
    main_config, create_config = module.main_config, module.create_config
    cfg_test = compile_config(main_config, seed=0, auto=True, create_cfg=create_config, save_cfg=False)
    assert cfg == cfg_test, 'cfg_formatted_failed'

import sys
from typing_extensions import Self
sys.path.append( '/home/PJLAB/chenyun/trade_test/DI-engine')
print(sys.path)
from typing import Union, Optional, List, Any, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.entry.utils import random_collect
from easydict import EasyDict
import json
from dizoo.trading_test.envs.stocks_env import StocksEnv


if __name__ == "__main__":
    print("ok")
    cfg = EasyDict({"env_id": [0], "eps_length": 2334, "window_size": 20})
    env = StocksEnv(cfg)
    env.reset(1)
    print(env.max_possible_profit())
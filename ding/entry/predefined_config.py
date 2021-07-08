from typing import Tuple
from easydict import EasyDict
import sys
import importlib

env_dict = {
    'cartpole': 'dizoo.classic_control.cartpole.config',
    'pendulum': 'dizoo.classic_control.pendulum.config',
}
policy_dict = {
    'dqn': 'ding.policy.dqn',
    'rainbow': 'ding.policy.rainbow',
    'c51': 'ding.policy.c51',
    'qrdqn': 'ding.policy.qrdqn',
    'iqn': 'ding.policy.iqn',
    'a2c': 'ding.policy.a2c',
    'impala': 'ding.policy.impala',
    'ppo': 'ding.policy.ppo',
    'sqn': 'ding.policy.sqn',
    'r2d2': 'ding.policy.r2d2',
    'ddpg': 'ding.policy.ddpg',
    'td3': 'ding.policy.td3',
    'sac': 'ding.policy.sac',
}


def get_predefined_config(env: str, policy: str) -> Tuple[EasyDict, EasyDict]:
    config_name = '{}_{}_config'.format(env, policy)
    create_config_name = '{}_{}_create_config'.format(env, policy)
    try:
        m = importlib.import_module(env_dict[env] + '.' + config_name)
        return [getattr(m, config_name), getattr(m, create_config_name)]
    except ImportError:
        print("Please get started by other types, there is no related pre-defined config({})".format(config_name))
        sys.exit(1)

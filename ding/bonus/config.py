from easydict import EasyDict
import gym
from ding.envs import BaseEnv, DingEnvWrapper
from ding.policy import PPOFPolicy


def get_instance_config(env: str) -> EasyDict:
    cfg = PPOFPolicy.default_config()
    if env == 'lunarlander_discrete':
        cfg.n_sample = 400
    elif env == 'lunarlander_continuous':
        cfg.action_space = 'continuous'
        cfg.n_sample = 400
    else:
        raise KeyError("not supported env type: {}".format(env))
    return cfg


def get_instance_env(env: str) -> BaseEnv:
    if env == 'lunarlander_discrete':
        return DingEnvWrapper(gym.make('LunarLander-v2'))
    elif env == 'lunarlander_continuous':
        return DingEnvWrapper(gym.make('LunarLander-v2', continuous=True))
    else:
        raise KeyError("not supported env type: {}".format(env))

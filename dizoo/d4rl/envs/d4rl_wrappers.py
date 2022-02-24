from typing import Dict
import gym
import numpy as np
import logging
from ding.envs import ObsNormWrapper, RewardNormWrapper

try:
    import d4rl  # register d4rl enviroments with open ai gym
except ImportError:
    logging.warning("not found d4rl env, please install it, refer to https://github.com/rail-berkeley/d4rl")


def wrap_d4rl(
        env_id,
        norm_obs: Dict = dict(use_norm=False, ),
        norm_reward: Dict = dict(use_norm=False, ),
        only_info=False
) -> gym.Env:
    r"""
    Overview:
        Wrap Mujoco Env to preprocess env step's return info, e.g. observation normalization, reward normalization, etc.
    Arguments:
        - env_id (:obj:`str`): Mujoco environment id, for example "HalfCheetah-v3"
        - norm_obs (:obj:`EasyDict`): Whether to normalize observation or not
        - norm_reward (:obj:`EasyDict`): Whether to normalize reward or not. For evaluator, environment's reward \
            should not be normalized: Either ``norm_reward`` is None or ``norm_reward.use_norm`` is False can do this.
    Returns:
        - wrapped_env (:obj:`gym.Env`): The wrapped mujoco environment
    """
    if not only_info:
        env = gym.make(env_id)
        if norm_obs is not None and norm_obs.use_norm:
            env = ObsNormWrapper(env)
        if norm_reward is not None and norm_reward.use_norm:
            env = RewardNormWrapper(env, norm_reward.reward_discount)
        return env
    else:
        wrapper_info = ''
        if norm_obs is not None and norm_obs.use_norm:
            wrapper_info = ObsNormWrapper.__name__ + '\n'
        if norm_reward is not None and norm_reward.use_norm:
            wrapper_info += RewardNormWrapper.__name__ + '\n'
        return wrapper_info

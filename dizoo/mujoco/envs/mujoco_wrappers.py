from typing import Dict
import gym
import numpy as np

from ding.envs import ObsNormWrapper, RewardNormWrapper, DelayRewardWrapper, FinalEvalRewardEnv


def wrap_mujoco(
        env_id,
        norm_obs: Dict = dict(use_norm=False, ),
        norm_reward: Dict = dict(use_norm=False, ),
        delay_reward_step: int = 1
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
    env = gym.make(env_id)
    env = FinalEvalRewardEnv(env)
    if norm_obs is not None and norm_obs.use_norm:
        env = ObsNormWrapper(env)
    if norm_reward is not None and norm_reward.use_norm:
        env = RewardNormWrapper(env, norm_reward.reward_discount)
    if delay_reward_step > 1:
        env = DelayRewardWrapper(env, delay_reward_step)

    return env

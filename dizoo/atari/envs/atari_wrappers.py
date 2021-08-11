# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import cv2
import gym
import numpy as np
from collections import deque
from ding.envs import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, \
                        ClipRewardEnv, FrameStack


def wrap_deepmind(
    env_id, episode_life=True, clip_rewards=True, frame_stack=4, scale=True, warp_frame=True, only_info=False
):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert 'NoFrameskip' in env_id
    if not only_info:
        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if episode_life:
            env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if warp_frame:
            env = WarpFrame(env)
        if scale:
            env = ScaledFloatFrame(env)
        if clip_rewards:
            env = ClipRewardEnv(env)
        if frame_stack:
            env = FrameStack(env, frame_stack)
        return env
    else:
        wrapper_info = NoopResetEnv.__name__ + '\n'
        wrapper_info += MaxAndSkipEnv.__name__ + '\n'
        if episode_life:
            wrapper_info += EpisodicLifeEnv.__name__ + '\n'
        # if 'FIRE' in env.unwrapped.get_action_meanings():
        if 'Pong' in env_id or 'Qbert' in env_id or 'SpaceInvader' in env_id or 'Montezuma' in env_id:
            wrapper_info += FireResetEnv.__name__ + '\n'
        if warp_frame:
            wrapper_info += WarpFrame.__name__ + '\n'
        if scale:
            wrapper_info += ScaledFloatFrame.__name__ + '\n'
        if clip_rewards:
            wrapper_info += ClipRewardEnv.__name__ + '\n'
        if frame_stack:
            wrapper_info += FrameStack.__name__ + '\n'
        return wrapper_info


def wrap_deepmind_mr(
    env_id, episode_life=True, clip_rewards=True, frame_stack=4, scale=True, warp_frame=True, only_info=False
):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert 'MontezumaReveng' in env_id
    if not only_info:
        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if episode_life:
            env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if warp_frame:
            env = WarpFrame(env)
        if scale:
            env = ScaledFloatFrame(env)
        if clip_rewards:
            env = ClipRewardEnv(env)
        if frame_stack:
            env = FrameStack(env, frame_stack)
        return env
    else:
        wrapper_info = NoopResetEnv.__name__ + '\n'
        wrapper_info += MaxAndSkipEnv.__name__ + '\n'
        if episode_life:
            wrapper_info += EpisodicLifeEnv.__name__ + '\n'
        # if 'FIRE' in env.unwrapped.get_action_meanings():
        if 'Pong' in env_id or 'Qbert' in env_id or 'SpaceInvader' in env_id or 'Montezuma' in env_id:
            wrapper_info += FireResetEnv.__name__ + '\n'
        if warp_frame:
            wrapper_info += WarpFrame.__name__ + '\n'
        if scale:
            wrapper_info += ScaledFloatFrame.__name__ + '\n'
        if clip_rewards:
            wrapper_info += ClipRewardEnv.__name__ + '\n'
        if frame_stack:
            wrapper_info += FrameStack.__name__ + '\n'
        return wrapper_info

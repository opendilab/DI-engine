# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import gym
from ding.envs import MaxAndSkipWrapper, RewardNormWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, \
    ClipRewardWrapper, FrameStackWrapper, ObsNormWrapper


def wrap_car_racing(env_id, episode_life=True, clip_rewards=True, frame_stack=4, scale=True, warp_frame=True,
                    obs_norm=True, rew_norm=True):
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
    assert 'CarRacing' in env_id
    env = gym.make(env_id)
    env = MaxAndSkipWrapper(env, skip=4)
    if warp_frame:
        env = WarpFrameWrapper(env)
    if scale:
        env = ScaledFloatFrameWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)
    if frame_stack:
        env = FrameStackWrapper(env, frame_stack)
    if obs_norm:
        env = ObsNormWrapper(env)
    # if rew_norm:
    #     env = RewardNormWrapper(env)
    return env

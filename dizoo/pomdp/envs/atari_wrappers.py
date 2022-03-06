# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import cv2
import gym
import numpy as np
from collections import deque
from copy import deepcopy
from torch import float32
import matplotlib.pyplot as plt

from ding.envs import RamWrapper, NoopResetWrapper, MaxAndSkipWrapper, EpisodicLifeWrapper, FireResetWrapper, WarpFrameWrapper, ClipRewardWrapper, FrameStackWrapper


class ScaledFloatFrameWrapper(gym.ObservationWrapper):
    """Normalize observations to -1~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(low=-1., high=1., shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # use fixed scale and bias temporarily
        return (observation - 128) / 128
        # return (observation - self.bias) / self.scale


class FrameStackWrapperRam(gym.Wrapper):
    """Stack n_frames last frames.
    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(
        self,
        env,
        n_frames,
        pomdp={
            "noise_scale": 0.01,
            "zero_p": 0.2,
            "duplicate_p": 0.2,
            "reward_noise": 0.01
        },
        render=False
    ):
        super().__init__(env)
        self.n_frames = n_frames
        self.n_dims = env.observation_space.shape[0]
        self._pomdp = pomdp
        self._render = render
        self.frames = deque([], maxlen=n_frames)
        self._images = deque([], maxlen=n_frames)
        self.viewer = None

        shape = (n_frames * self.n_dims, )
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
            _img = self.env.unwrapped._get_image()
            _img = _img.mean(axis=-1, keepdims=True).astype(np.uint8)
            self._images.append(_img)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        reward = reward + self._pomdp["reward_noise"] * np.random.randn()

        if self._render:
            _img = self.env.unwrapped._get_image()
            _img = _img.mean(axis=-1, keepdims=True).astype(np.uint8)
            self._images.append(_img)
            self.render()

        return self._get_ob(), reward, done, info

    def render(self):
        from gym.envs.classic_control import rendering
        state = np.stack(self._images, axis=0)
        obs = self._pomdp_preprocess(state, img=True).astype(np.uint8)
        obs = np.tile(obs[-1], (1, 1, 3))
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(obs)
        return self.viewer.isopen

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        state = np.stack(self.frames, axis=0)
        obs = self._pomdp_preprocess(state)

        return obs.flatten()

    def _pomdp_preprocess(self, state, img=False):
        obs = deepcopy(state)
        # POMDP process
        if np.random.random() > (1 - self._pomdp["duplicate_p"]):
            update_end_point = np.random.randint(
                1, self.n_frames
            )  # choose a point from that point we can't get new observation
            _s = (self.n_frames - update_end_point, 1, 1, 1)
            obs[update_end_point:, ] = np.tile(obs[update_end_point, ], _s)

        if img:
            pomdp_noise_mask = self._pomdp["noise_scale"] * np.random.randn(*obs.shape) * 128
        else:
            pomdp_noise_mask = self._pomdp["noise_scale"] * np.random.randn(*obs.shape)

        # Flickering Atari game
        obs = obs * int(np.random.random() > self._pomdp["zero_p"]) + pomdp_noise_mask
        return obs.astype(np.float32)


def wrap_deepmind(
    env_id,
    episode_life=True,
    clip_rewards=True,
    pomdp={},
    frame_stack=4,
    scale=True,
    warp_frame=True,
    use_ram=False,
    render=False,
    only_info=False
):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :param float pomdp: parameter to control POMDP prepropress,
    :return: the wrapped atari environment.
    """
    assert 'NoFrameskip' in env_id
    if not only_info:
        env = gym.make(env_id)
        env = RamWrapper(env)
        env = NoopResetWrapper(env, noop_max=30)
        env = MaxAndSkipWrapper(env, skip=4)
        if episode_life:
            env = EpisodicLifeWrapper(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetWrapper(env)
        if warp_frame:
            env = WarpFrameWrapper(env)
        if scale:
            env = ScaledFloatFrameWrapper(env)
        if clip_rewards:
            env = ClipRewardWrapper(env)

        if frame_stack:
            if use_ram:
                env = FrameStackWrapperRam(env, frame_stack, pomdp, render)
            else:
                env = FrameStackWrapper(env, frame_stack)

        return env
    else:
        wrapper_info = RamWrapper.__name__ + '\n'
        wrapper_info += NoopResetWrapper.__name__ + '\n'
        wrapper_info += MaxAndSkipWrapper.__name__ + '\n'
        if episode_life:
            wrapper_info = EpisodicLifeWrapper.__name__ + '\n'
        if 'Pong' in env_id or 'Qbert' in env_id or 'SpaceInvader' in env_id or 'Montezuma' in env_id:
            wrapper_info = FireResetWrapper.__name__ + '\n'
        if warp_frame:
            wrapper_info = WarpFrameWrapper.__name__ + '\n'
        if scale:
            wrapper_info = ScaledFloatFrameWrapper.__name__ + '\n'
        if clip_rewards:
            wrapper_info = ClipRewardWrapper.__name__ + '\n'

        if frame_stack:
            if use_ram:
                wrapper_info = FrameStackWrapperRam.__name__ + '\n'
            else:
                wrapper_info = FrameStackWrapper.__name__ + '\n'

        return wrapper_info

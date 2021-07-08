# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import cv2
import gym
import os.path as osp
import numpy as np
from typing import Union, Optional
from collections import deque
from copy import deepcopy
from torch import float32
# import matplotlib.pyplot as plt


class NoopResetEnv(gym.Wrapper):
    """
    Overview:
       Sample initial states by taking random number of no-ops on reset.  \
       No-op is assumed to be action 0.
    Interface:
        ``__init__``, ``reset``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - noop_max (:obj:`int`): the maximum value of no-ops to run.
    """

    def __init__(self, env, noop_max=30):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - noop_max (:obj:`int`): the maximum value of no-ops to run.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """
        Overview:
            Resets the state of the environment and returns an initial observation.
        Returns:
            - observation (:obj:`Any`): the initial observation.
        """
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)

        .. note::
            Shapes space such as\
                observation space shape, could be some nested strcutures, \
                but for some simple environments, like pong in Atari, \
                    observation space shape is a ``tuple``——``(4, 84, 84)``.
            Same case applies for the ``new_shape`` interface in the following wrapper classes.
        """
        return obs_shape, act_shape, rew_shape


class MaxAndSkipEnv(gym.Wrapper):
    """
    Overview:
       Return only every `skip`-th frame (frameskipping) using most  \
       recent raw observations (for max pooling across time steps)
    Interface:
        ``__init__``, ``step``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - skip (:obj:`int`): number of `skip`-th frame.

    """

    def __init__(self, env, skip=4):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - skip (:obj:`int`): number of `skip`-th frame.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action,  \
                sum reward, and max over last observations.
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - max_frame (:obj:`np.array`) : max over last observations
            - total_reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further step()  \
                calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful for  \
                debugging, and sometimes learning)

        """
        obs_list, total_reward, done = [], 0., False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            obs_list.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        return max_frame, total_reward, done, info

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class WarpFrame(gym.ObservationWrapper):
    """
    Overview:
       Warp frames to 84x84 as done in the Nature paper and later work.
    Interface:
        ``__init__``, ``observation``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``size=84``, ``obs_space``,  ``self.observation_space``

    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.size = 84
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.tuple.Tuple):
            obs_space = (obs_space, )
        self.observation_space = gym.spaces.tuple.Tuple(
            [
                gym.spaces.Box(
                    low=np.min(obs_space[0].low),
                    high=np.max(obs_space[0].high),
                    shape=(self.size, self.size),
                    dtype=obs_space[0].dtype
                ) for _ in range(len(obs_space))
            ]
        )
        if len(self.observation_space) == 1:
            self.observation_space = self.observation_space[0]

    def observation(self, frame):
        """
        Overview:
            Returns the current observation from a frame
        Arguments:
            - frame (:obj:`Any`): the frame to get observation from
        Returns:
            - observation (:obj:`Any`): Framed observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
            Get new shape of observation, acton, and reward; in this case only  \
                observation space wrapped to (4, 84, 84); others unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return (4, 84, 84), act_shape, rew_shape


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Overview:
       Normalize observations to 0~1.
    Interface:
        ``__init__``, ``observation``, ``new_shape``
    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        """
        Overview:
            Returns the scaled observation
        Arguments:
            - observation(:obj:`Float`): The original observation
        Returns:
            - observation (:obj:`Float`): The Scaled Float observation
        """

        return ((observation - self.bias) / self.scale).astype('float32')

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
            Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class ClipRewardEnv(gym.RewardWrapper):
    """
    Overview:
       clips the reward to {+1, 0, -1} by its sign.
    Interface:
        ``__init__``, ``reward``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``reward_range``

    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """
        Overview:
            Bin reward to {+1, 0, -1} by its sign. Note: np.sign(0) == 0.
        Arguments:
            - reward(:obj:`Float`): Raw Reward
        Returns:
            - reward(:obj:`Float`): Clipped Reward
        """
        return np.sign(reward)

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class FrameStack(gym.Wrapper):
    """
    Overview:
       Stack n_frames last frames.
    Interface:
        ``__init__``, ``reset``, ``step``, ``_get_ob``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - n_frame (:obj:`int`): the number of frames to stack.
        - ``observation_space``, ``frames``
    """

    def __init__(self, env, n_frames):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - n_frame (:obj:`int`): the number of frames to stack.
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.tuple.Tuple):
            obs_space = (obs_space, )
        shape = (n_frames, ) + obs_space[0].shape
        self.observation_space = gym.spaces.tuple.Tuple(
            [
                gym.spaces.Box(
                    low=np.min(obs_space[0].low), high=np.max(obs_space[0].high), shape=shape, dtype=obs_space[0].dtype
                ) for _ in range(len(obs_space))
            ]
        )
        if len(self.observation_space) == 1:
            self.observation_space = self.observation_space[0]

    def reset(self):
        """
        Overview:
            Resets the state of the environment and append new observation to frames
        Returns:
            - ``self._get_ob()``: observation
        """
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action, sum reward,  \
                and max over last observations, and append new observation to frames
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - ``self._get_ob()`` : observation
            - reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further \
                 step() calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful  \
                for debugging, and sometimes learning)
        """

        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        """
        Overview:
            The original wrapper use `LazyFrames` but since we use np buffer, it has no effect
        """
        return np.stack(self.frames, axis=0)

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
            Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class ObsTransposeWrapper(gym.ObservationWrapper):
    """
    Overview:
       Wrapper to transpose env, usually used in atari environments
    Interface:
        ``__init__``, ``observation``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``observation_space``
    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature;  \
                setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.tuple.Tuple):
            self.observation_space = gym.spaces.Box(
                low=np.min(obs_space[0].low),
                high=np.max(obs_space[0].high),
                shape=(len(obs_space), obs_space[0].shape[2], obs_space[0].shape[0], obs_space[0].shape[1]),
                dtype=obs_space[0].dtype
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.min(obs_space.low),
                high=np.max(obs_space.high),
                shape=(obs_space.shape[2], obs_space.shape[0], obs_space.shape[1]),
                dtype=obs_space.dtype
            )

    def observation(self, obs: Union[tuple, np.ndarray]):
        """
        Overview:
            Returns the transposed observation
        Arguments:
            - observation(:obj:`Union[tuple, np.ndarray]`): The original observation
        Returns:
            - observation (:obj:`Union[tuple, np.ndarray]`): The transposed observation
        """
        if isinstance(obs, tuple):
            new_obs = []
            for i in range(len(obs)):
                new_obs.append(obs[i].transpose(2, 0, 1))
            obs = np.stack(new_obs)
        else:
            obs = obs.transpose(2, 0, 1)
        return obs

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
            Get new shape of observation, acton, and reward; in this case with  \
                obs_space transposed and others unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        if isinstance(obs_shape, tuple):
            new_obs = []
            for i in range(len(obs_shape)):
                new_obs.append(obs_shape[i].transpose(2, 0, 1))
            obs_shape = np.stack(new_obs)
        else:
            obs_shape = obs_shape.transpose(2, 0, 1)
        return obs_shape, act_shape, rew_shape


class RunningMeanStd(object):
    """
    Overview:
       Wrapper to update new variable, new mean, and new count
    Interface:
        ``__init__``, ``update``, ``reset``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.

        - ``mean``, ``std``, ``_epsilon``, ``_shape``, ``_mean``, ``_var``, ``_count``
    """

    def __init__(self, epsilon=1e-4, shape=()):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate  \
                signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - epsilon (:obj:`Float`): the epsilon used for self for the std output
            - shape (:obj: `np.array`): the np array shape used for the expression  \
                of this wrapper on attibutes of mean and variance
        """
        self._epsilon = epsilon
        self._shape = shape
        self.reset()

    def update(self, x):
        """
        Overview:
            Update mean, variable, and count
        Arguments:
            - ``x``: the batch
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = batch_count + self._count
        mean_delta = batch_mean - self._mean
        new_mean = self._mean + mean_delta * batch_count / new_count
        # this method for calculating new variable might be numerically unstable
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
        new_var = m2 / new_count
        self._mean = new_mean
        self._var = new_var
        self._count = new_count

    def reset(self):
        """
        Overview:
            Resets the state of the environment and reset properties:  \
                ``_mean``, ``_var``, ``_count``
        """
        self._mean = np.zeros(self._shape, 'float64')
        self._var = np.ones(self._shape, 'float64')
        self._count = self._epsilon

    @property
    def mean(self) -> np.ndarray:
        """
        Overview:
            Property ``mean`` gotten  from ``self._mean``
        """
        return self._mean

    @property
    def std(self) -> np.ndarray:
        """
        Overview:
            Property ``std`` calculated  from ``self._var`` and the epsilon value of ``self._epsilon``
        """
        return np.sqrt(self._var) + self._epsilon

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class ObsNormEnv(gym.ObservationWrapper):
    """
    Overview:
       Normalize observations according to running mean and std.
    Interface:
        ``__init__``, ``step``, ``reset``, ``observation``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.

        - ``data_count``, ``clip_range``, ``rms``
    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature;  \
                setup the properties according to running mean and std.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.data_count = 0
        self.clip_range = (-3, 3)
        self.rms = RunningMeanStd(shape=env.observation_space.shape)

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action, sum reward,  \
                and update ``data_count``, and also update the ``self.rms`` property  \
                    once after integrating with the input ``action``.
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - ``self.observation(observation)`` : normalized observation after the  \
                input action and updated ``self.rms``
            - reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further  \
                step() calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful  \
                for debugging, and sometimes learning)

        """
        self.data_count += 1
        observation, reward, done, info = self.env.step(action)
        self.rms.update(observation)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        """
        Overview:
            Get obeservation
        Arguments:
            - observation (:obj:`Any`): Original observation
        Returns:
            - observation (:obj:`Any`): Normalized new observation

        """
        if self.data_count > 30:
            return np.clip((observation - self.rms.mean) / self.rms.std, self.clip_range[0], self.clip_range[1])
        else:
            return observation

    def reset(self, **kwargs):
        """
        Overview:
            Resets the state of the environment and reset properties.
        Arguments:
            - kwargs (:obj:`Dict`): Reset with this key argumets
        Returns:
            - observation (:obj:`Any`): New observation after reset

        """
        self.data_count = 0
        self.rms.reset()
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class RewardNormEnv(gym.RewardWrapper):
    """
    Overview:
       Normalize reward according to running std.
    Interface:
        ``__init__``, ``step``, ``reward``, ``reset``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``cum_reward``, ``reward_discount``, ``data_count``, ``rms``
    """

    def __init__(self, env, reward_discount):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature;  \
                setup the properties according to running mean and std.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.cum_reward = np.zeros((1, ), 'float64')
        self.reward_discount = reward_discount
        self.data_count = 0
        self.rms = RunningMeanStd(shape=(1, ))

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action, sum reward,  \
                and update ``data_count``, and also update the ``self.rms`` and ``self.cum_reward``  \
                    properties once after integrating with the input ``action``.
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - observation : normalized observation after the input action and updated ``self.rms``
            - ``self.reward(reward)`` : amount of reward returned after previous action \
                 (normalized) and update ``self.cum_reward``
            - done (:obj:`Bool`) : whether the episode has ended, in which case further  \
                step() calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful for \
                debugging, and sometimes learning)

        """
        self.data_count += 1
        observation, reward, done, info = self.env.step(action)
        reward = np.array([reward], 'float64')
        self.cum_reward = self.cum_reward * self.reward_discount + reward
        self.rms.update(self.cum_reward)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        """
        Overview:
           Normalize reward if ``data_count`` is more than 30
        Arguments:
            - reward(:obj:`Float`): Raw Reward
        Returns:
            - reward(:obj:`Float`): Normalized Reward
        """
        if self.data_count > 30:
            return float(reward / self.rms.std)
        else:
            return float(reward)

    def reset(self, **kwargs):
        """
        Overview:
            Resets the state of the environment and reset properties (`NumType` ones to 0, \
                and ``self.rms`` as reset rms wrapper)
        Arguments:
            - kwargs (:obj:`Dict`): Reset with this key argumets
        """
        self.cum_reward = 0.
        self.data_count = 0
        self.rms.reset()
        return self.env.reset(**kwargs)

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class RamWrapper(gym.Wrapper):
    """
    Overview:
       Wrapper ram env into image-like env
    Interface:
        ``__init__``, ``reset``, ``step``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - n_frame (:obj:`int`): the number of frames to stack.
        - ``observation_space``
    """

    def __init__(self, env, render=False):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature;
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        shape = env.observation_space.shape + (1, 1)
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=np.float32
        )

    def reset(self):
        """
        Overview:
            Resets the state of the environment and reset properties.

        Returns:
            - observation (:obj:`Any`): New observation after reset and reshaped

        """
        obs = self.env.reset()
        return obs.reshape(128, 1, 1).astype(np.float32)

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action, sum reward and \
                reshape the observation.
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - ``obs.reshape(128, 1, 1).astype(np.float32)`` : reshaped observation after \
                step with type restriction.
            - reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further \
                step() calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful for \
                debugging, and sometimes learning)

        """
        obs, reward, done, info = self.env.step(action)
        return obs.reshape(128, 1, 1).astype(np.float32), reward, done, info

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
            Get new shape of observation, acton, and reward; in this case only observation \
                space wrapped to (128,1,1); others unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return (128, 1, 1), act_shape, rew_shape


class EpisodicLifeEnv(gym.Wrapper):
    """
    Overview:
        Make end-of-life == end-of-episode, but only reset on true game over. It helps \
            the value estimation.
    Interface:
        ``__init__``, ``step``, ``reset``, ``observation``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.

        - ``lives``, ``was_real_done``
    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; set \
                lives to 0 at set done.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action, sum reward; set \
                ``self.was_real_done`` as done, and step according to lives i.e. check \
                    current lives, make loss of life terminal, then update lives to \
                        handle bonus lives.
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - obs (:obj:`Any`): normalized observation after the input action and updated ``self.rms``
            - reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further step() \
                calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful for debugging,\
                 and sometimes learning)

        """
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """
        Overview:
            Calls the Gym environment reset, only when lives are exhausted. This way all states are \
                still reachable even though lives are episodic, and the learner need not know about \
                    any of this behind-the-scenes.
        Returns:
            - obs (:obj:`Any`): New observation after reset with no-op step to advance from terminal/lost \
                life state in case of not ``self.was_real_done``.

        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs = self.env.step(0)[0]
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class FireResetEnv(gym.Wrapper):
    """
    Overview:
        Take action on reset for environments that are fixed until firing. \
            Related discussion: https://github.com/openai/baselines/issues/240
    Interface:
        ``__init__``, ``reset``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        """
        Overview:
            Resets the state of the environment and reset properties i.e. reset with action 1
        """
        self.env.reset()
        return self.env.step(1)[0]

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


def update_shape(obs_shape, act_shape, rew_shape, wrapper_names):
    """
    Overview:
        Get new shape of observation, acton, and reward given the wrapper.
    Arguments:
        obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`), wrapper_names (:obj:`Any`)
    Returns:
        obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
    """
    for wrapper_name in wrapper_names:
        if wrapper_name:
            try:
                obs_shape, act_shape, rew_shape = eval(wrapper_name).new_shape(obs_shape, act_shape, rew_shape)
            except:
                continue
    return obs_shape, act_shape, rew_shape

"""
This code is adapted from OpenAI Baselines:
    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

List of Environment Wrappers:
- NoopResetWrapper: This wrapper facilitates the sampling of initial states by executing a random number of
    no-operation actions upon environment reset.
- MaxAndSkipWrapper: Incorporates max pooling across time steps, a method that reduces the temporal dimension by taking
    the maximum value over specified time intervals.
- WarpFrameWrapper: Implements frame warping by resizing the images to 84x84, a common preprocessing step in
    reinforcement learning on visual data, as described in the DeepMind Nature paper and subsequent works.
- ScaledFloatFrameWrapper: Normalizes observations to a range of 0 to 1, which is a common requirement for neural
    network inputs.
- ClipRewardWrapper: Clips the reward to {-1, 0, +1} based on its sign. This simplifies the reward structure and
    can make learning more stable in environments with high variance in rewards.
- DelayRewardWrapper: Returns cumulative reward at defined intervals, and at all other times, returns a reward of 0.
    This can be useful for sparse reward problems.
- FrameStackWrapper: Stacks the latest 'n' frames as a single observation. This allows the agent to have a sense of
    dynamics and motion from the stacked frames.
- ObsTransposeWrapper: Transposes the observation to bring the channel to the first dimension, a common requirement
    for convolutional neural networks.
- ObsNormWrapper: Normalizes observations based on a running mean and standard deviation. This can help to standardize
    inputs for the agent and speed up learning.
- RewardNormWrapper: Normalizes reward based on a running standard deviation, which can stabilize learning in
    environments with high variance in rewards.
- RamWrapper: Wraps a RAM-based environment into an image-like environment. This can be useful for applying
    image-based algorithms to RAM-based Atari games.
- EpisodicLifeWrapper: Treats end of life as the end of an episode, but only resets on true game over. This can help
    the agent better differentiate between losing a life and losing the game.
- FireResetWrapper: Executes the 'fire' action upon environment reset. This is specific to certain Atari games where
    the 'fire' action starts the game.
- GymHybridDictActionWrapper: Transforms the original `gym.spaces.Tuple` action space into a `gym.spaces.Dict`.
- FlatObsWrapper: Flattens image and language observations into a single vector, which can be helpful for input into
    certain types of models.
- StaticObsNormWrapper: Provides functionality for normalizing observations according to a static mean and
    standard deviation.
- EvalEpisodeReturnWrapper: Evaluates the return over an episode during evaluation, providing a more comprehensive
    view of the agent's performance.
- GymToGymnasiumWrapper: Adapts environments from the Gym library to be compatible with the Gymnasium library.
- AllinObsWrapper: Consolidates all information into the observation, useful for environments where the agent's
    observation should include additional information such as the current score or time remaining.
- ObsPlusPrevActRewWrapper: This wrapper is used in policy NGU. It sets a dict as the new wrapped observation,
    which includes the current observation, previous action and previous reward.
"""

import copy
import operator
from collections import deque
from functools import reduce
from typing import Union, Any, Tuple, Dict, List

import gym
import gymnasium
import numpy as np
from easydict import EasyDict

from ding.torch_utils import to_ndarray
from ding.utils import ENV_WRAPPER_REGISTRY, import_module


@ENV_WRAPPER_REGISTRY.register('noop_reset')
class NoopResetWrapper(gym.Wrapper):
    """
    Overview:
       Sample initial states by taking random number of no-ops on reset.  No-op is assumed to be action 0.
    Interfaces:
        __init__, reset
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - noop_max (:obj:`int`): the maximum value of no-ops to run.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        """
        Overview:
            Initialize the NoopResetWrapper.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - noop_max (:obj:`int`): the maximum value of no-ops to run. Defaults to 30.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and returns an initial observation,
            after taking a random number of no-ops.
        Returns:
            - observation (:obj:`Any`): The initial observation after no-ops.
        """
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs


@ENV_WRAPPER_REGISTRY.register('max_and_skip')
class MaxAndSkipWrapper(gym.Wrapper):
    """
    Overview:
       Wraps the environment to return only every ``skip``-th frame (frameskipping) \
       using most recent raw observations (for max pooling across time steps).
    Interfaces:
        __init__, step
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
        - skip (:obj:`int`): Number of ``skip``-th frame. Defaults to 4.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        """
        Overview:
            Initialize the MaxAndSkipWrapper.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - skip (:obj:`int`): Number of ``skip``-th frame. Defaults to 4.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action: Union[int, np.ndarray]) -> tuple:
        """
        Overview:
            Take the given action and repeat it for a specified number of steps. \
            The rewards are summed up and the maximum frame over the last observations is returned.
        Arguments:
            - action (:obj:`Any`): The action to repeat.
        Returns:
            - max_frame (:obj:`np.array`): Max over last observations
            - total_reward (:obj:`Any`): Sum of rewards after previous action.
            - done (:obj:`Bool`): Whether the episode has ended.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for  \
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


@ENV_WRAPPER_REGISTRY.register('warp_frame')
class WarpFrameWrapper(gym.ObservationWrapper):
    """
    Overview:
        The WarpFrameWrapper class is a gym observation wrapper that resizes
        the frame of an environment observation to a specified size (default is 84x84).
        This is often used in the preprocessing pipeline of observations in reinforcement learning,
        especially for visual observations from Atari environments.
    Interfaces:
        __init__, observation
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - size (:obj:`int`): the size to which the frames are to be resized.
        - observation_space (:obj:`gym.Space`): the observation space of the wrapped environment.
    """

    def __init__(self, env: gym.Env, size: int = 84):
        """
        Overview:
            Constructor for WarpFrameWrapper class, initializes the environment and the size.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - size (:obj:`int`): the size to which the frames are to be resized. Default is 84.
        """
        super().__init__(env)
        self.size = size
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

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        Overview:
            Resize the frame (observation) to the desired size.
        Arguments:
            - frame (:obj:`np.ndarray`): the frame to be resized.
        Returns:
            - frame (:obj:`np.ndarray`): the resized frame.
        """
        try:
            import cv2
        except ImportError:
            from ditk import logging
            import sys
            logging.warning("Please install opencv-python first.")
            sys.exit(1)
        # deal with the `channel_first` case
        if frame.shape[0] < 10:
            frame = frame.transpose(1, 2, 0)
            frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
            frame = frame.transpose(2, 0, 1)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return frame


@ENV_WRAPPER_REGISTRY.register('scaled_float_frame')
class ScaledFloatFrameWrapper(gym.ObservationWrapper):
    """
    Overview:
        The ScaledFloatFrameWrapper normalizes observations to between 0 and 1.
    Interfaces:
        __init__, observation
    """

    def __init__(self, env: gym.Env):
        """
        Overview:
            Initialize the ScaledFloatFrameWrapper, setting the scale and bias for normalization.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Overview:
            Scale the observation to be within the range [0, 1].
        Arguments:
            - observation (:obj:`np.ndarray`): the original observation.
        Returns:
            - scaled_observation (:obj:`np.ndarray`): the scaled observation.
        """
        return ((observation - self.bias) / self.scale).astype('float32')


@ENV_WRAPPER_REGISTRY.register('clip_reward')
class ClipRewardWrapper(gym.RewardWrapper):
    """
    Overview:
        The ClipRewardWrapper class is a gym reward wrapper that clips the reward to {-1, 0, +1} based on its sign.
        This can be used to normalize the scale of the rewards in reinforcement learning algorithms.
    Interfaces:
        __init__, reward
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - reward_range (:obj:`Tuple[int, int]`): the range of the reward values after clipping.
    """

    def __init__(self, env: gym.Env):
        """
        Overview:
            Initialize the ClipRewardWrapper class.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward: float) -> float:
        """
        Overview:
            Clip the reward to {-1, 0, +1} based on its sign. Note: np.sign(0) == 0.
        Arguments:
            - reward (:obj:`float`): the original reward.
        Returns:
            - reward (:obj:`float`): the clipped reward.
        """
        return np.sign(reward)


@ENV_WRAPPER_REGISTRY.register('action_repeat')
class ActionRepeatWrapper(gym.Wrapper):
    """
    Overview:
        The ActionRepeatWrapper class is a gym wrapper that repeats the same action for a number of steps.
        This wrapper is particularly useful in environments where the desired effect is achieved by maintaining
        the same action across multiple time steps. For instance, some physical environments like motion control
        tasks might require consistent force input to produce a significant state change.

        Using this wrapper can reduce the temporal complexity of the problem, as it allows the agent to perform
        multiple actions within a single time step. This can speed up learning, as the agent has fewer decisions
        to make within a time step. However, it may also sacrifice some level of decision-making precision, as the
        agent cannot change its action across successive time steps.

        Note that the use of the ActionRepeatWrapper may not be suitable for all types of environments. Specifically,
        it may not be the best choice for environments where new decisions must be made at each time step, or where
        the time sequence of actions has a significant impact on the outcome.
    Interfaces:
        __init__, step
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - action_repeat (:obj:`int`): the number of times to repeat the action.
    """

    def __init__(self, env: gym.Env, action_repeat: int = 1):
        """
        Overview:
            Initialize the ActionRepeatWrapper class.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - action_repeat (:obj:`int`): the number of times to repeat the action. Default is 1.
        """
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action: Union[int, np.ndarray]) -> tuple:
        """
        Overview:
            Take the given action and repeat it for a specified number of steps. The rewards are summed up.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): The action to repeat.
        Returns:
            - obs (:obj:`np.ndarray`): The observation after repeating the action.
            - reward (:obj:`float`): The sum of rewards after repeating the action.
            - done (:obj:`bool`): Whether the episode has ended.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information.
        """
        reward = 0
        for _ in range(self.action_repeat):
            obs, rew, done, info = self.env.step(action)
            reward += rew or 0
            if done:
                break
        return obs, reward, done, info


@ENV_WRAPPER_REGISTRY.register('delay_reward')
class DelayRewardWrapper(gym.Wrapper):
    """
    Overview:
        The DelayRewardWrapper class is a gym wrapper that delays the reward. It cumulates the reward over a
        predefined number of steps and returns the cumulated reward only at the end of this interval.
        At other times, it returns a reward of 0.

        This wrapper is particularly useful in environments where the impact of an action is not immediately
        observable, but rather delayed over several steps. For instance, in strategic games or planning tasks,
        the effect of an action may not be directly noticeable, but it contributes to a sequence of actions that
        leads to a reward. In these cases, delaying the reward to match the action-effect delay can make the
        learning process more consistent with the problem's nature.

        However, using this wrapper may increase the difficulty of learning, as the agent needs to associate its
        actions with delayed outcomes. It also introduces a non-standard reward structure, which could limit the
        applicability of certain reinforcement learning algorithms.

        Note that the use of the DelayRewardWrapper may not be suitable for all types of environments. Specifically,
        it may not be the best choice for environments where the effect of actions is immediately observable and the
        reward should be assigned accordingly.
    Interfaces:
        __init__, reset, step
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - delay_reward_step (:obj:`int`): the number of steps over which to delay and cumulate the reward.
    """

    def __init__(self, env: gym.Env, delay_reward_step: int = 0):
        """
        Overview:
            Initialize the DelayRewardWrapper class.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - delay_reward_step (:obj:`int`): the number of steps over which to delay and cumulate the reward.
        """
        super().__init__(env)
        self._delay_reward_step = delay_reward_step

    def reset(self) -> np.ndarray:
        """
         Overview:
             Resets the state of the environment and resets the delay reward duration and current delay reward.
         Returns:
             - obs (:obj:`np.ndarray`): the initial observation of the environment.
         """
        self._delay_reward_duration = 0
        self._current_delay_reward = 0.
        obs = self.env.reset()
        return obs

    def step(self, action: Union[int, np.ndarray]) -> tuple:
        """
        Overview:
            Take the given action and repeat it for a specified number of steps. The rewards are summed up.
            If the number of steps equals the delay reward step, return the cumulated reward and reset the
            delay reward duration and current delay reward. Otherwise, return a reward of 0.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): the action to take in the step.
        Returns:
            - obs (:obj:`np.ndarray`): The observation after the step.
            - reward (:obj:`float`): The cumulated reward after the delay reward step or 0.
            - done (:obj:`bool`): Whether the episode has ended.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information.
        """
        obs, reward, done, info = self.env.step(action)
        self._current_delay_reward += reward
        self._delay_reward_duration += 1
        if done or self._delay_reward_duration >= self._delay_reward_step:
            reward = self._current_delay_reward
            self._current_delay_reward = 0.
            self._delay_reward_duration = 0
        else:
            reward = 0.
        return obs, reward, done, info


@ENV_WRAPPER_REGISTRY.register('eval_episode_return')
class EvalEpisodeReturnWrapper(gym.Wrapper):
    """
    Overview:
        A wrapper for a gym environment that accumulates rewards at every timestep, and returns the total reward at the
        end of the episode in `info`. This is used for evaluation purposes.
    Interfaces:
        __init__, reset, step
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        """
        Overview:
            Initialize the EvalEpisodeReturnWrapper. This involves setting up the environment to wrap.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)

    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment and initialize the accumulated reward to zero.
        Returns:
            - obs (:obj:`np.ndarray`): The initial observation from the environment.
        """
        self._eval_episode_return = 0.
        return self.env.reset()

    def step(self, action: Any) -> tuple:
        """
        Overview:
            Step the environment with the provided action, accumulate the returned reward, and add the total reward to
            `info` if the episode is done.
        Arguments:
            - action (:obj:`Any`): The action to take in the environment.
        Returns:
            - obs (:obj:`np.ndarray`): The next observation from the environment.
            - reward (:obj:`float`): The reward from taking the action.
            - done (:obj:`bool`): Whether the episode is done.
            - info (:obj:`Dict[str, Any]`): A dictionary of extra information, which includes 'eval_episode_return' if
                the episode is done.
        Examples:
            >>> env = gym.make("CartPole-v1")
            >>> env = EvalEpisodeReturnWrapper(env)
            >>> obs = env.reset()
            >>> done = False
            >>> while not done:
            ...     action = env.action_space.sample()  # Replace with your own policy
            ...     obs, reward, done, info = env.step(action)
            ...     if done:
            ...         print("Total episode reward:", info['eval_episode_return'])
        """
        obs, reward, done, info = self.env.step(action)
        self._eval_episode_return += reward
        if done:
            info['eval_episode_return'] = to_ndarray([self._eval_episode_return], dtype=np.float32)
        return obs, reward, done, info


@ENV_WRAPPER_REGISTRY.register('frame_stack')
class FrameStackWrapper(gym.Wrapper):
    """
     Overview:
        FrameStackWrapper is a gym environment wrapper that stacks the latest n frames (generally 4 in Atari)
        as a single observation. It is commonly used in environments where the observation is an image,
        and consecutive frames provide useful temporal information for the agent.
     Interfaces:
         __init__, reset, step, _get_ob
     Properties:
         - env (:obj:`gym.Env`): The environment to wrap.
         - n_frames (:obj:`int`): The number of frames to stack.
         - frames (:obj:`collections.deque`): A queue that holds the most recent frames.
         - observation_space (:obj:`gym.Space`): The space of the stacked observations.
     """

    def __init__(self, env: gym.Env, n_frames: int = 4) -> None:
        """
        Overview:
            Initialize the FrameStackWrapper. This process includes setting up the environment to wrap,
            the number of frames to stack, and the observation space.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - n_frame (:obj:`int`): The number of frames to stack.
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

    def reset(self) -> np.ndarray:
        """
        Overview:
            Reset the environment and initialize frames with the initial observation.
        Returns:
            - init_obs (:obj:`np.ndarray`): The stacked initial observations.
        """
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Overview:
            Perform a step in the environment with the given action, append the returned observation
            to frames, and return the stacked observations.
        Arguments:
            - action (:obj:`Any`): The action to perform a step with.
        Returns:
            - self._get_ob() (:obj:`np.ndarray`): The stacked observations.
            - reward (:obj:`float`): The amount of reward returned after the previous action.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
              undefined results.
            - info (:obj:`Dict[str, Any]`): Contains auxiliary diagnostic information (helpful for debugging,
              and sometimes learning).
        """
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self) -> np.ndarray:
        """
        Overview:
            The original wrapper used `LazyFrames`, but since we use an np buffer, it has no effect.
        Returns:
            - stacked_frames (:obj:`np.ndarray`): The stacked frames.
        """
        return np.stack(self.frames, axis=0)


@ENV_WRAPPER_REGISTRY.register('obs_transpose')
class ObsTransposeWrapper(gym.ObservationWrapper):
    """
    Overview:
        The ObsTransposeWrapper class is a gym wrapper that transposes the observation to put the channel dimension
        first. This can be helpful for certain types of neural networks that expect the channel dimension to be
        the first dimension.
    Interfaces:
        __init__, observation
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
        - observation_space (:obj:`gym.spaces.Box`): The transformed observation space.
    """

    def __init__(self, env: gym.Env):
        """
        Overview:
            Initialize the ObsTransposeWrapper class and update the observation space according to the environment's
            observation space.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
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

    def observation(self, obs: Union[tuple, np.ndarray]) -> Union[tuple, np.ndarray]:
        """
        Overview:
            Transpose the observation to put the channel dimension first. If the observation is a tuple, each element
            in the tuple is transposed independently.
        Arguments:
            - obs (:obj:`Union[tuple, np.ndarray]`): The original observation.
        Returns:
            - obs (:obj:`Union[tuple, np.ndarray]`): The transposed observation.
        """
        if isinstance(obs, tuple):
            new_obs = []
            for i in range(len(obs)):
                new_obs.append(obs[i].transpose(2, 0, 1))
            obs = np.stack(new_obs)
        else:
            obs = obs.transpose(2, 0, 1)
        return obs


class RunningMeanStd(object):
    """
    Overview:
       The RunningMeanStd class is a utility that maintains a running mean and standard deviation calculation over
        a stream of data.
    Interfaces:
        __init__, update, reset, mean, std
    Properties:
        - mean (:obj:`np.ndarray`): The running mean.
        - std (:obj:`np.ndarray`): The running standard deviation.
        - _epsilon (:obj:`float`): A small number to prevent division by zero when calculating standard deviation.
        - _shape (:obj:`tuple`): The shape of the data stream.
        - _mean (:obj:`np.ndarray`): The current mean of the data stream.
        - _var (:obj:`np.ndarray`): The current variance of the data stream.
        - _count (:obj:`float`): The number of data points processed.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Overview:
            Initialize the RunningMeanStd object.
        Arguments:
            - epsilon (:obj:`float`, optional): A small number to prevent division by zero when calculating standard
                deviation. Default is 1e-4.
            - shape (:obj:`tuple`, optional): The shape of the data stream. Default is an empty tuple, which
                corresponds to scalars.
        """
        self._epsilon = epsilon
        self._shape = shape
        self.reset()

    def update(self, x: np.array):
        """
        Overview:
            Update the running statistics with a new batch of data.
        Arguments:
            - x (:obj:`np.array`): A batch of data.
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
            Get the current running mean.
        Returns:
            The current running mean.
        """
        return self._mean

    @property
    def std(self) -> np.ndarray:
        """
        Overview:
            Get the current running standard deviation.
        Returns:
            The current running mean.
        """
        return np.sqrt(self._var) + self._epsilon


@ENV_WRAPPER_REGISTRY.register('obs_norm')
class ObsNormWrapper(gym.ObservationWrapper):
    """
    Overview:
        The ObsNormWrapper class is a gym observation wrapper that normalizes
        observations according to running mean and standard deviation (std).
    Interfaces:
        __init__, step, reset, observation
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - data_count (:obj:`int`): the count of data points observed so far.
        - clip_range (:obj:`Tuple[int, int]`): the range to clip the normalized observation.
        - rms (:obj:`RunningMeanStd`): running mean and standard deviation of the observations.
    """

    def __init__(self, env: gym.Env):
        """
        Overview:
            Initialize the ObsNormWrapper class.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.data_count = 0
        self.clip_range = (-3, 3)
        self.rms = RunningMeanStd(shape=env.observation_space.shape)

    def step(self, action: Union[int, np.ndarray]):
        """
        Overview:
            Take an action in the environment, update the running mean and std,
            and return the normalized observation.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): the action to take in the environment.
        Returns:
            - obs (:obj:`np.ndarray`): the normalized observation after the action.
            - reward (:obj:`float`): the reward after the action.
            - done (:obj:`bool`): whether the episode has ended.
            - info (:obj:`Dict`): contains auxiliary diagnostic information.
        """
        self.data_count += 1
        observation, reward, done, info = self.env.step(action)
        self.rms.update(observation)
        return self.observation(observation), reward, done, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalize the observation using the current running mean and std.
            If less than 30 data points have been observed, return the original observation.
        Arguments:
            - observation (:obj:`np.ndarray`): the original observation.
        Returns:
            - observation (:obj:`np.ndarray`): the normalized observation.
        """
        if self.data_count > 30:
            return np.clip((observation - self.rms.mean) / self.rms.std, self.clip_range[0], self.clip_range[1])
        else:
            return observation

    def reset(self, **kwargs):
        """
        Overview:
            Reset the environment and the properties related to the running mean and std.
        Arguments:
            - kwargs (:obj:`Dict`): keyword arguments to be passed to the environment's reset function.
        Returns:
            - observation (:obj:`np.ndarray`): the initial observation of the environment.
        """
        self.data_count = 0
        self.rms.reset()
        observation = self.env.reset(**kwargs)
        return self.observation(observation)


@ENV_WRAPPER_REGISTRY.register('static_obs_norm')
class StaticObsNormWrapper(gym.ObservationWrapper):
    """
    Overview:
        The StaticObsNormWrapper class is a gym observation wrapper that normalizes
        observations according to a precomputed mean and standard deviation (std) from a fixed dataset.
    Interfaces:
        __init__, observation
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - mean (:obj:`numpy.ndarray`): the mean of the observations in the fixed dataset.
        - std (:obj:`numpy.ndarray`): the standard deviation of the observations in the fixed dataset.
        - clip_range (:obj:`Tuple[int, int]`): the range to clip the normalized observation.
    """

    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray):
        """
        Overview:
            Initialize the StaticObsNormWrapper class.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - mean (:obj:`numpy.ndarray`): the mean of the observations in the fixed dataset.
            - std (:obj:`numpy.ndarray`): the standard deviation of the observations in the fixed dataset.
        """
        super().__init__(env)
        self.mean = mean
        self.std = std
        self.clip_range = (-3, 3)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalize the given observation using the precomputed mean and std.
            The normalized observation is then clipped within the specified range.
        Arguments:
            - observation (:obj:`np.ndarray`): the original observation.
        Returns:
            - observation (:obj:`np.ndarray`): the normalized and clipped observation.
        """
        return np.clip((observation - self.mean) / self.std, self.clip_range[0], self.clip_range[1])


@ENV_WRAPPER_REGISTRY.register('reward_norm')
class RewardNormWrapper(gym.RewardWrapper):
    """
    Overview:
        This wrapper class normalizes the reward according to running std. It extends the `gym.RewardWrapper`.
    Interfaces:
        __init__, step, reward, reset
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
        - cum_reward (:obj:`numpy.ndarray`): The cumulated reward, initialized as zero and updated in `step` method.
        - reward_discount (:obj:`float`): The discount factor for reward.
        - data_count (:obj:`int`): A counter for data, incremented in each `step` call.
        - rms (:obj:`RunningMeanStd`): An instance of RunningMeanStd to compute the running mean and std of reward.
    """

    def __init__(self, env: gym.Env, reward_discount: float) -> None:
        """
        Overview:
            Initialize the RewardNormWrapper, setup the properties according to running mean and std.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - reward_discount (:obj:`float`): The discount factor for reward.
        """
        super().__init__(env)
        self.cum_reward = np.zeros((1, ), 'float64')
        self.reward_discount = reward_discount
        self.data_count = 0
        self.rms = RunningMeanStd(shape=(1, ))

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Overview:
            Step the environment with the given action, update properties and return the new observation, reward,
            done status and info.
        Arguments:
            - action (:obj:`Any`): The action to execute in the environment.
        Returns:
            - observation (:obj:`np.ndarray`): Normalized observation after executing the action and updated `self.rms`.
            - reward (:obj:`float`): Amount of reward returned after the action execution (normalized) and updated
                `self.cum_reward`.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
                undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and sometimes
                learning).
        """
        self.data_count += 1
        observation, reward, done, info = self.env.step(action)
        reward = np.array([reward], 'float64')
        self.cum_reward = self.cum_reward * self.reward_discount + reward
        self.rms.update(self.cum_reward)
        return observation, self.reward(reward), done, info

    def reward(self, reward: float) -> float:
        """
        Overview:
            Normalize reward if `data_count` is more than 30.
        Arguments:
            - reward (:obj:`float`): The raw reward.
        Returns:
            - reward (:obj:`float`): Normalized reward.
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


@ENV_WRAPPER_REGISTRY.register('ram')
class RamWrapper(gym.Wrapper):
    """
    Overview:
        This wrapper class wraps a RAM environment into an image-like environment. It extends the `gym.Wrapper`.
    Interfaces:
        __init__, reset, step
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
        - observation_space (:obj:`gym.spaces.Box`): The observation space of the wrapped environment.
    """

    def __init__(self, env: gym.Env, render: bool = False) -> None:
        """
        Overview:
            Initialize the RamWrapper and set up the observation space to wrap the RAM environment.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - render (:obj:`bool`): Whether to render the environment, default is False.
        """
        super().__init__(env)
        shape = env.observation_space.shape + (1, 1)
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and returns a reshaped observation.
        Returns:
            - observation (:obj:`np.ndarray`): New observation after reset and reshaped.
        """
        obs = self.env.reset()
        return obs.reshape(128, 1, 1).astype(np.float32)

    def step(self, action: Any) -> Tuple[np.ndarray, Any, bool, Dict]:
        """
        Overview:
            Execute one step within the environment with the given action. Repeat action, sum reward and reshape the
            observation.
        Arguments:
            - action (:obj:`Any`): The action to take in the environment.
        Returns:
            - observation (:obj:`np.ndarray`): Reshaped observation after step with type restriction.
            - reward (:obj:`Any`): Amount of reward returned after previous action.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
              undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and sometimes
              learning).
        """
        obs, reward, done, info = self.env.step(action)
        return obs.reshape(128, 1, 1).astype(np.float32), reward, done, info


@ENV_WRAPPER_REGISTRY.register('episodic_life')
class EpisodicLifeWrapper(gym.Wrapper):
    """
    Overview:
        This wrapper makes end-of-life equivalent to end-of-episode, but only resets on
        true game over. This helps in better value estimation.
    Interfaces:
        __init__, step, reset
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
        - lives (:obj:`int`): The current number of lives.
        - was_real_done (:obj:`bool`): Whether the last episode was ended due to game over.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Overview:
            Initialize the EpisodicLifeWrapper, setting lives to 0 and was_real_done to True.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Overview:
            Execute the given action in the environment, update properties based on the new
            state and return the new observation, reward, done status and info.
        Arguments:
            - action (:obj:`Any`): The action to execute in the environment.
        Returns:
            - observation (:obj:`np.ndarray`): Normalized observation after the action execution and updated `self.rms`.
            - reward (:obj:`float`): Amount of reward returned after the action execution.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
                undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and
                sometimes learning).
        """
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # For Qbert sometimes we stay in lives == 0 condition for a few frames,
            # so it is important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and updates the number of lives, only when
            lives are exhausted. This way all states are still reachable even though lives
            are episodic, and the learner need not know about any of this behind-the-scenes.
        Returns:
            - observation (:obj:`np.ndarray`): New observation after reset with no-op step to advance from
                terminal/lost life state.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs = self.env.step(0)[0]
        self.lives = self.env.unwrapped.ale.lives()
        return obs


@ENV_WRAPPER_REGISTRY.register('fire_reset')
class FireResetWrapper(gym.Wrapper):
    """
    Overview:
        This wrapper takes a fire action at environment reset.
        Related discussion: https://github.com/openai/baselines/issues/240
    Interfaces:
        __init__, reset
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Overview:
            Initialize the FireResetWrapper. Assume that the second action of the environment
            is 'FIRE' and there are at least three actions.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and executes a fire action, i.e. reset with action 1.
        Returns:
            - observation (:obj:`np.ndarray`): New observation after reset and fire action.
        """
        self.env.reset()
        return self.env.step(1)[0]


@ENV_WRAPPER_REGISTRY.register('gym_hybrid_dict_action')
class GymHybridDictActionWrapper(gym.ActionWrapper):
    """
    Overview:
        Transform Gym-Hybrid's original `gym.spaces.Tuple` action space to `gym.spaces.Dict`.
    Interfaces:
        __init__, action
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
        - action_space (:obj:`gym.spaces.Dict`): The new action space.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Overview:
            Initialize the GymHybridDictActionWrapper, setting up the new action space.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)
        self.action_space = gym.spaces.Dict(
            {
                'type': gym.spaces.Discrete(3),
                # shape = (2, )  0 is for acceleration; 1 is for rotation
                'mask': gym.spaces.Box(low=0, high=1, shape=(2, ), dtype=np.int64),
                'args': gym.spaces.Box(
                    low=np.array([0., -1.], dtype=np.float32),
                    high=np.array([1., 1.], dtype=np.float32),
                    shape=(2, ),
                    dtype=np.float32
                ),
            }
        )

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        Overview:
            Execute the given action in the environment, transform the action from Dict to Tuple,
            and return the new observation, reward, done status and info.
        Arguments:
            - action (:obj:`Dict`): The action to execute in the environment, structured as a dictionary.
        Returns:
            - observation (:obj:`Dict`): The wrapped observation, which includes the current observation,
                previous action and previous reward.
            - reward (:obj:`float`): Amount of reward returned after the action execution.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
                undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and
                sometimes learning).
        """
        # # From Dict to Tuple
        # action_type = action[0]
        # if action_type == 0:
        #     action_mask = np.array([1, 0], dtype=np.int64)
        #     action_args = np.array([action[1][0], 0], dtype=np.float32)
        # elif action_type == 1:
        #     action_mask = np.array([0, 1], dtype=np.int64)
        #     action_args = np.array([0, action[1][1]], dtype=np.float32)
        # elif action_type == 2:
        #     action_mask = np.array([0, 0], dtype=np.int64)
        #     action_args = np.array([0, 0], dtype=np.float32)

        # From Dict to Tuple
        action_type, action_mask, action_args = action['type'], action['mask'], action['args']
        return self.env.step((action_type, action_args))


@ENV_WRAPPER_REGISTRY.register('obs_plus_prev_action_reward')
class ObsPlusPrevActRewWrapper(gym.Wrapper):
    """
    Overview:
        This wrapper is used in policy NGU. It sets a dict as the new wrapped observation,
        which includes the current observation, previous action and previous reward.
    Interfaces:
        __init__, reset, step
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
        - prev_action (:obj:`int`): The previous action.
        - prev_reward_extrinsic (:obj:`float`): The previous reward.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Overview:
            Initialize the ObsPlusPrevActRewWrapper, setting up the previous action and reward.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                'obs': env.observation_space,
                'prev_action': env.action_space,
                'prev_reward_extrinsic': gym.spaces.Box(
                    low=env.reward_range[0], high=env.reward_range[1], shape=(1, ), dtype=np.float32
                )
            }
        )
        self.prev_action = -1  # null action
        self.prev_reward_extrinsic = 0  # null reward

    def reset(self) -> Dict:
        """
        Overview:
            Resets the state of the environment, and returns the wrapped observation.
        Returns:
            - observation (:obj:`Dict`): The wrapped observation, which includes the current observation,
                previous action and previous reward.
        """
        obs = self.env.reset()
        obs = {'obs': obs, 'prev_action': self.prev_action, 'prev_reward_extrinsic': self.prev_reward_extrinsic}
        return obs

    def step(self, action: Any) -> Tuple[Dict, float, bool, Dict]:
        """
        Overview:
            Execute the given action in the environment, save the previous action and reward
            to be used in the next observation, and return the new observation, reward,
            done status and info.
        Arguments:
            - action (:obj:`Any`): The action to execute in the environment.
        Returns:
            - observation (:obj:`Dict`): The wrapped observation, which includes the current observation,
                previous action and previous reward.
            - reward (:obj:`float`): Amount of reward returned after the action execution.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
                undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and sometimes
                learning).
        """
        obs, reward, done, info = self.env.step(action)
        obs = {'obs': obs, 'prev_action': self.prev_action, 'prev_reward_extrinsic': self.prev_reward_extrinsic}
        self.prev_action = action
        self.prev_reward_extrinsic = reward
        return obs, reward, done, info


class TransposeWrapper(gym.Wrapper):
    """
    Overview:
        This class is used to transpose the observation space of the environment.

    Interfaces:
        __init__, _process_obs, step, reset
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Overview:
            Initialize the TransposeWrapper, setting up the new observation space.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)
        old_space = copy.deepcopy(env.observation_space)
        new_shape = (old_space.shape[-1], *old_space.shape[:-1])
        self._observation_space = gym.spaces.Box(
            low=old_space.low.min(), high=old_space.high.max(), shape=new_shape, dtype=old_space.dtype
        )

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Overview:
            Transpose the observation into the format (channels, height, width).
        Arguments:
            - obs (:obj:`np.ndarray`): The observation to transform.
        Returns:
            - obs (:obj:`np.ndarray`): The transposed observation.
        """
        obs = to_ndarray(obs)
        obs = np.transpose(obs, (2, 0, 1))
        return obs

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Overview:
            Execute the given action in the environment, process the observation and return
            the new observation, reward, done status, and info.
        Arguments:
            - action (:obj:`Any`): The action to execute in the environment.
        Returns:
            - observation (:obj:`np.ndarray`): The processed observation after the action execution.
            - reward (:obj:`float`): Amount of reward returned after the action execution.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
                undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and sometimes
                learning).
        """
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def reset(self) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and returns the processed observation.
        Returns:
            - observation (:obj:`np.ndarray`): The processed observation after reset.
        """
        obs = self.env.reset()
        return self._process_obs(obs)


class TimeLimitWrapper(gym.Wrapper):
    """
    Overview:
        This class is used to enforce a time limit on the environment.
    Interfaces:
        __init__, reset, step
    """

    def __init__(self, env: gym.Env, max_limit: int) -> None:
        """
        Overview:
            Initialize the TimeLimitWrapper, setting up the maximum limit of time steps.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - max_limit (:obj:`int`): The maximum limit of time steps.
        """
        super().__init__(env)
        self.max_limit = max_limit

    def reset(self) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and the time counter.
        Returns:
            - observation (:obj:`np.ndarray`): The new observation after reset.
        """
        self.time_count = 0
        return self.env.reset()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Overview:
            Execute the given action in the environment, update the time counter, and
            return the new observation, reward, done status and info.
        Arguments:
            - action (:obj:`Any`): The action to execute in the environment.
        Returns:
            - observation (:obj:`np.ndarray`): The new observation after the action execution.
            - reward (:obj:`float`): Amount of reward returned after the action execution.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
                undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and sometimes
                learning).
        """
        obs, reward, done, info = self.env.step(action)
        self.time_count += 1
        if self.time_count >= self.max_limit:
            done = True
            info['time_limit'] = True
        else:
            info['time_limit'] = False
        info['time_count'] = self.time_count
        return obs, reward, done, info


class FlatObsWrapper(gym.Wrapper):
    """
    Overview:
        This class is used to flatten the observation space of the environment.
        Note: only suitable for environments like minigrid.
    Interfaces:
        __init__, observation, reset, step
    """

    def __init__(self, env: gym.Env, maxStrLen: int = 96) -> None:
        """
        Overview:
            Initialize the FlatObsWrapper, setup the new observation space.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - maxStrLen (:obj:`int`): The maximum length of mission string, default is 96.
        """
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen, ),
            dtype="float32",
        )

        self.cachedStr: str = None

    def observation(self, obs: Union[np.ndarray, Tuple]) -> np.ndarray:
        """
        Overview:
            Process the observation, convert the mission into one-hot encoding and concatenate
            it with the image data.
        Arguments:
            - obs (:obj:`Union[np.ndarray, Tuple]`): The raw observation to process.
        Returns:
            - obs (:obj:`np.ndarray`): The processed observation.
        """
        if isinstance(obs, tuple):  # for compatibility of gymnasium
            obs = obs[0]
        image = obs["image"]
        mission = obs["mission"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (len(mission) <= self.maxStrLen), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype="float32")

            for idx, ch in enumerate(mission):
                if ch >= "a" and ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(f"Character {ch} is not available in mission string.")
                assert chNo < self.numCharCodes, "%s : %d" % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

    def reset(self, *args, **kwargs) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and returns the processed observation.
        Returns:
            - observation (:obj:`np.ndarray`): The processed observation after reset.
        """
        obs = self.env.reset(*args, **kwargs)
        return self.observation(obs)

    def step(self, *args, **kwargs) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Overview:
            Execute the given action in the environment, and return the processed observation,
            reward, done status, and info.
        Returns:
            - observation (:obj:`np.ndarray`): The processed observation after the action execution.
            - reward (:obj:`float`): Amount of reward returned after the action execution.
            - done (:obj:`bool`): Whether the episode has ended, in which case further step() calls will return
                undefined results.
            - info (:obj:`Dict`): Contains auxiliary diagnostic information (helpful for debugging, and sometimes
                learning).
        """
        o, r, d, i = self.env.step(*args, **kwargs)
        o = self.observation(o)
        return o, r, d, i


class GymToGymnasiumWrapper(gym.Wrapper):
    """
    Overview:
        This class is used to wrap a gymnasium environment to a gym environment.
    Interfaces:
        __init__, seed, reset, step
    """

    def __init__(self, env: gymnasium.Env) -> None:
        """
        Overview:
            Initialize the GymToGymnasiumWrapper.
        Arguments:
            - env (:obj:`gymnasium.Env`): The gymnasium environment to wrap.
        """
        assert isinstance(env, gymnasium.Env), type(env)
        super().__init__(env)
        self._seed = None

    def seed(self, seed: int) -> None:
        """
        Overview:
            Set the seed for the environment.
        Arguments:
            - seed (:obj:`int`): The seed to set.
        """
        self._seed = seed

    def reset(self) -> np.ndarray:
        """
        Overview:
            Resets the state of the environment and returns the new observation. If a seed
            was set, use it in the reset.
        Returns:
            - observation (:obj:`np.ndarray`): The new observation after reset.
        """
        if self.seed is not None:
            obs, info = self.env.reset(seed=self._seed)
        else:
            obs, info = self.env.reset()
        return obs

    def step(self, *args, **kwargs):
        """
        Overview:
            Execute the given action in the environment, and return the new observation,
            reward, done status, and info. To keep consistency with gym, the done status should be the either \
            terminated=True or truncated=True.
        """
        obs, rew, terminated, truncated, info = self.env.step(*args, **kwargs)
        return obs, rew, terminated or truncated, info


@ENV_WRAPPER_REGISTRY.register('reward_in_obs')
class AllinObsWrapper(gym.Wrapper):
    """
    Overview:
        This wrapper is used in policy ``Decision Transformer``, which is proposed in paper
        https://arxiv.org/abs/2106.01345. It sets a dict {'obs': obs, 'reward': reward}
        as the new wrapped observation, which includes the current observation and previous reward.
    Interfaces:
        __init__, reset, step, seed
    Properties:
        - env (:obj:`gym.Env`): The environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Overview:
            Initialize the AllinObsWrapper.
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)

    def reset(self) -> Dict:
        """
        Overview:
            Resets the state of the environment and returns the new observation.
        Returns:
            - observation (:obj:`Dict`): The new observation after reset, includes the current observation and reward.
        """
        ret = {'obs': self.env.reset(), 'reward': np.array([0])}
        self._observation_space = gym.spaces.Dict(
            {
                'obs': self.env.observation_space,
                'reward': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(1, ))
            }
        )
        return ret

    def step(self, action: Any):
        """
        Overview:
            Execute the given action in the environment, and return the new observation,
            reward, done status, and info.
        Arguments:
            - action (:obj:`Any`): The action to execute in the environment.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): The timestep after the action execution.
        """
        obs, reward, done, info = self.env.step(action)
        obs = {'obs': obs, 'reward': reward}
        from ding.envs import BaseEnvTimestep
        return BaseEnvTimestep(obs, reward, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Overview:
            Set the seed for the environment.
        Arguments:
            - seed (:obj:`int`): The seed to set.
            - dynamic_seed (:obj:`bool`): Whether to use dynamic seed, default is True.
        """
        self.env.seed(seed, dynamic_seed)


def update_shape(obs_shape: Any, act_shape: Any, rew_shape: Any, wrapper_names: List[str]) -> Tuple[Any, Any, Any]:
    """
    Overview:
        Get new shapes of observation, action, and reward given the wrapper.
    Arguments:
        - obs_shape (:obj:`Any`): The original shape of observation.
        - act_shape (:obj:`Any`): The original shape of action.
        - rew_shape (:obj:`Any`): The original shape of reward.
        - wrapper_names (:obj:`List[str]`): The names of the wrappers.
    Returns:
        - obs_shape (:obj:`Any`): The new shape of observation.
        - act_shape (:obj:`Any`): The new shape of action.
        - rew_shape (:obj:`Any`): The new shape of reward.
    """
    for wrapper_name in wrapper_names:
        if wrapper_name:
            try:
                obs_shape, act_shape, rew_shape = eval(wrapper_name).new_shape(obs_shape, act_shape, rew_shape)
            except Exception:
                continue
    return obs_shape, act_shape, rew_shape


def create_env_wrapper(env: gym.Env, env_wrapper_cfg: EasyDict) -> gym.Wrapper:
    """
    Overview:
        Create an environment wrapper according to the environment wrapper configuration and the environment instance.
    Arguments:
        - env (:obj:`gym.Env`): The environment instance to be wrapped.
        - env_wrapper_cfg (:obj:`EasyDict`): The configuration for the environment wrapper.
    Returns:
        - env (:obj:`gym.Wrapper`): The wrapped environment instance.
    """
    env_wrapper_cfg = copy.deepcopy(env_wrapper_cfg)
    if 'import_names' in env_wrapper_cfg:
        import_module(env_wrapper_cfg.pop('import_names'))
    env_wrapper_type = env_wrapper_cfg.pop('type')
    return ENV_WRAPPER_REGISTRY.build(env_wrapper_type, env, **env_wrapper_cfg.get('kwargs', {}))

from typing import Any, List, Union, Optional
import time
import gym
import numpy as np

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray

from ding.worker.collector.tests.speed_test.utils import random_change

global env_sum
env_sum = 0


def env_sleep(duration):
    time.sleep(duration)
    global env_sum
    env_sum += duration


class FakeEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._obs_dim = cfg.get('obs_dim', 4)
        self._action_dim = cfg.get('action_dim', 2)
        self._episode_step_base = cfg.get('episode_step', 200)
        self._reset_time = cfg.get('reset_time', 0.)
        self._step_time = cfg.get('step_time', 0.)
        self.reset()
        # gym attribute
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 1}
        self._observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim, ), dtype=np.float32)
        self._action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(self._action_dim, ), dtype=np.float32)
        self._reward_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
        self._init_flag = True

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed'):
            self.seed()
        self._episode_step = int(random_change(self._episode_step_base))
        env_sleep(random_change(self._reset_time))
        self._step_count = 0
        self._final_eval_reward = 0.
        obs = np.random.randn(self._obs_dim).astype(np.float32)
        return obs

    def close(self) -> None:
        self._init_flag = False

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        env_sleep(random_change(self._step_time))
        self._step_count += 1
        obs = np.random.randn(self._obs_dim).astype(np.float32)
        rew = np.random.randint(2)
        done = True if self._step_count == self._episode_step else False
        info = {}
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        rew = to_ndarray([rew])  # to shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        return "DI-engine Fake Env for collector profile test"

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

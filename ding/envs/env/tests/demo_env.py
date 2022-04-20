from typing import Any, Union
import gym
import numpy as np

from ding.envs.env import BaseEnv, BaseEnvTimestep


class DemoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._closed = True
        # It is highly recommended to implement these three spaces
        self._observation_space = gym.spaces.Dict(
            {
                "demo_dict": gym.spaces.Tuple(
                    [
                        gym.spaces.Box(low=-10., high=10., shape=(4, ), dtype=np.float32),
                        gym.spaces.Box(low=-100., high=100., shape=(1, ), dtype=np.float32)
                    ]
                )
            }
        )
        self._action_space = gym.spaces.Discrete(5)
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def reset(self) -> Any:
        """
        Overview:
            Resets the env to an initial state and returns an initial observation. Abstract Method from ``gym.Env``.
        """
        self._step_count = 0
        self._env = "A real environment"
        self._closed = False
        return self.observation_space.sample()

    def close(self) -> None:
        self._closed = True

    def step(self, action: Any) -> 'BaseEnv.timestep':
        self._step_count += 1
        obs = self.observation_space.sample()
        rew = self.reward_space.sample()
        if self._step_count == 30:
            self._step_count = 0
            done = True
        else:
            done = False
        info = {}
        if done:
            info['final_eval_reward'] = self.reward_space.sample() * 30
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int) -> None:
        self._seed = seed

    def random_action(self) -> Union[np.ndarray, int]:
        return self.action_space.sample()

    def __repr__(self) -> str:
        return "Demo Env for env_implementation_test.py"

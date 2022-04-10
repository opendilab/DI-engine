from typing import Any
import gym
import numpy as np
import pytest

from ding.envs.env import BaseEnv, BaseEnvTimestep
from ding.envs.env.env_implementation_check import check_reset, check_step, check_obs_deepcopy


class DemoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._is_closed = False
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
        return self.observation_space.sample()

    def close(self) -> None:
        self._is_closed = True

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

    def __repr__(self) -> str:
        return "Demo Env for env_implementation_test.py"


@pytest.mark.unittest
def test_an_implemented_env():
    demo_env = DemoEnv({})
    check_reset(demo_env)
    check_step(demo_env)
    check_obs_deepcopy(demo_env)

from typing import Union, Optional
import gym
from gym.spaces import Box
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
import dmc2gym


def dmc2gym_observation_space(dim, minimum=-np.inf, maximum=np.inf, dytpe=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dytpe), np.repeat(maximum, dim).astype(dytpe), dtype=dytpe)


def dmc2gym_state_space(dim, minimum=-np.inf, maximum=np.inf, dytpe=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dytpe), np.repeat(maximum, dim).astype(dytpe), dtype=dytpe)


def dmc2gym_action_space(dim, minimum=-1, maximum=1, dytpe=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dytpe), np.repeat(maximum, dim).astype(dytpe), dtype=dytpe)


def dmc2gym_reward_space(minimum=0, maximum=1) -> tuple:
    return (
        minimum,
        maximum,
    )


dmc2gym_env_info = {
    "ball_in_cup": {
        "catch": {
            "observation_space": dmc2gym_observation_space(8),
            "state_space": dmc2gym_state_space(8),
            "action_space": dmc2gym_action_space(2),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "cartpole": {
        "balance": {
            "observation_space": dmc2gym_observation_space(8),
            "state_space": dmc2gym_state_space(8),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        },
        "swingup": {
            "observation_space": dmc2gym_observation_space(8),
            "state_space": dmc2gym_state_space(8),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "cheetah": {
        "run": {
            "observation_space": dmc2gym_observation_space(17),
            "state_space": dmc2gym_state_space(17),
            "action_space": dmc2gym_action_space(6),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "finger": {
        "spin": {
            "observation_space": dmc2gym_observation_space(9),
            "state_space": dmc2gym_state_space(9),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "reacher": {
        "easy": {
            "observation_space": dmc2gym_observation_space(6),
            "state_space": dmc2gym_state_space(6),
            "action_space": dmc2gym_action_space(2),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "walker": {
        "walk": {
            "observation_space": dmc2gym_observation_space(24),
            "state_space": dmc2gym_state_space(24),
            "action_space": dmc2gym_action_space(6),
            "reward_space": dmc2gym_reward_space()
        }
    }
}


@ENV_REGISTRY.register('dmc2gym')
class DMC2GymEnv(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:
        assert cfg.domain_name in dmc2gym_env_info
        assert cfg.task_name in dmc2gym_env_info[cfg.domain_name]
        self._domain_name = cfg.domain_name
        self._task_name = cfg.task_name

        self._cfg = cfg
        self._init_flag = False

        #to do
        self._replay_path = None

        self._observation_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["observation_space"]
        self._action_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["action_space"]
        self._reward_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["reward_space"]

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = dmc2gym.make(domain_name=self._domain_name, task_name=self._task_name)
            #To do
            if self._replay_path is not None:
                pass
            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        #self._observation_space = self._env.observation_space
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        action = action.astype('float32')
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)  # wrapped to be transfered to a array with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.float32)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Deepmind Control Suite to gym Env: " + self._domain_name + ":" + self._task_name

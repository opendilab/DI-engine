from typing import Optional, Callable
import gym
from gym.spaces import Box
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
import dmc2gym


def dmc2gym_observation_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def dmc2gym_state_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def dmc2gym_action_space(dim, minimum=-1, maximum=1, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def dmc2gym_reward_space(minimum=0, maximum=1, dtype=np.float32) -> Callable:

    def reward_space(frame_skip=1) -> Box:
        return Box(
            np.repeat(minimum * frame_skip, 1).astype(dtype),
            np.repeat(maximum * frame_skip, 1).astype(dtype),
            dtype=dtype
        )

    return reward_space


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
        assert cfg.domain_name in dmc2gym_env_info, '{}/{}'.format(cfg.domain_name, dmc2gym_env_info.keys())
        assert cfg.task_name in dmc2gym_env_info[
            cfg.domain_name], '{}/{}'.format(cfg.task_name, dmc2gym_env_info[cfg.domain_name].keys())

        self._domain_name = cfg.domain_name
        self._task_name = cfg.task_name
        if "frame_skip" in cfg:
            self._frame_skip = cfg.frame_skip
        else:
            self._frame_skip = 1
        if "from_pixels" in cfg:
            self._from_pixels = cfg.from_pixels
        else:
            self._from_pixels = False
        if "visualize_reward" in cfg:
            self._visualize_reward = cfg.visualize_reward
        else:
            self._visualize_reward = False
        if "height" in cfg:
            self._height = cfg.height
        else:
            self._height = 84
        if "width" in cfg:
            self._width = cfg.width
        else:
            self._width = 84

        self._cfg = cfg
        self._init_flag = False

        # TODO(zjow) Add replay configuration.
        self._replay_path = None

        self._observation_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["observation_space"]
        self._action_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["action_space"]
        self._reward_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["reward_space"](self._frame_skip)

    def reset(self) -> np.ndarray:
        if not self._init_flag:

            self._env = dmc2gym.make(
                domain_name=self._domain_name,
                task_name=self._task_name,
                seed=1,
                visualize_reward=self._visualize_reward,
                from_pixels=self._from_pixels,
                height=self._height,
                width=self._width,
                frame_skip=self._frame_skip
            )

            #TODO(zjow) Add replay recording.
            if self._replay_path is not None:
                pass

            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

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

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
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
        random_action = self.action_space.sample().astype(np.float32)
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

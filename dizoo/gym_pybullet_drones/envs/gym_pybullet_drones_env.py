from typing import Optional, Callable
import gym
from gym.spaces import Box
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

import gym_pybullet_drones

# from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
# from gym_pybullet_drones.utils.utils import sync, str2bool

def gym_pybullet_drones_observation_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Callable:
    lower_bound=np.repeat(minimum, dim).astype(dtype)
    upper_bound=np.repeat(maximum, dim).astype(dtype)
    lower_bound[2]=0.0
    return Box(lower_bound, upper_bound, dtype=dtype)

def gym_pybullet_drones_action_space(dim, minimum=-1, maximum=1, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)

def gym_pybullet_drones_reward_space(minimum=-10000, maximum=0, dtype=np.float32) -> Callable:
    return Box(
        np.repeat(minimum, 1).astype(dtype),
        np.repeat(maximum, 1).astype(dtype),
        dtype=dtype
    )

gym_pybullet_drones_env_info={
    "takeoff-aviary-v0": {
        "observation_space": gym_pybullet_drones_observation_space(12,minimum=-1,maximum=1),
        "action_space": gym_pybullet_drones_action_space(4,minimum=-1,maximum=1),
        "reward_space": gym_pybullet_drones_reward_space()
    },
}

@ENV_REGISTRY.register('gym_pybullet_drones')
class GymPybulletDronesEnv(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:

        self._cfg = cfg
        self._env_id = cfg.env_id
        self._init_flag = False
        self._replay_path = None
        self._observation_space = gym_pybullet_drones_env_info[cfg.env_id]["observation_space"]
        self._action_space = gym_pybullet_drones_env_info[cfg.env_id]["action_space"]
        self._reward_space = gym_pybullet_drones_env_info[cfg.env_id]["reward_space"]

    def reset(self) -> np.ndarray:
        if not self._init_flag:

            self._env = gym.make(self._env_id)

            if self._replay_path is not None:
                if gym.version.VERSION > '0.22.0':
                    self._env.metadata.update({'render_modes': ["rgb_array"]})
                else:
                    self._env.metadata.update({'render.modes': ["rgb_array"]})
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )
                self._env.start_video_recorder()

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
        if not self._init_flag:
            return self._observation_space
        else:
            return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        if not self._init_flag:
            return self._action_space
        else:
            return self._env.action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space


    def __repr__(self) -> str:
        return "DI-engine gym_pybullet_drones Env: " + self._cfg["env_id"]

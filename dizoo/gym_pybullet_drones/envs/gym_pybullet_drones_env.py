from typing import Optional, Callable
import gym
from gym.spaces import Box
import numpy as np

import gym_pybullet_drones
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.utils.utils import sync, str2bool


def gym_pybullet_drones_observation_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Callable:
    lower_bound = np.repeat(minimum, dim).astype(dtype)
    upper_bound = np.repeat(maximum, dim).astype(dtype)
    lower_bound[2] = 0.0
    return Box(lower_bound, upper_bound, dtype=dtype)


def gym_pybullet_drones_action_space(dim, minimum=-1, maximum=1, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def gym_pybullet_drones_reward_space(minimum=-10000, maximum=0, dtype=np.float32) -> Callable:
    return Box(np.repeat(minimum, 1).astype(dtype), np.repeat(maximum, 1).astype(dtype), dtype=dtype)


gym_pybullet_drones_env_info = {
    "takeoff-aviary-v0": {
        "observation_space": gym_pybullet_drones_observation_space(12, minimum=-1, maximum=1),
        "action_space": gym_pybullet_drones_action_space(4, minimum=-1, maximum=1),
        "reward_space": gym_pybullet_drones_reward_space()
    },
}


@ENV_REGISTRY.register('gym_pybullet_drones')
class GymPybulletDronesEnv(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:

        if "num_drones" not in cfg:
            self.env_kwargs = {
                'drone_model': DroneModel.CF2X,
                'initial_xyzs': None,
                'initial_rpys': None,
                'physics': Physics.PYB,
                'freq': 240,
                'aggregate_phy_steps': 1,
                'gui': False,
                'record': False,
                'obs': ObservationType.KIN,
                'act': ActionType.RPM
            }
        else:
            #TODO for multi drones envs
            self.env_kwargs = {
                'drone_model': DroneModel.CF2X,
                'num_drones': 2,
                'neighbourhood_radius': np.inf,
                'initial_xyzs': None,
                'initial_rpys': None,
                'physics': Physics.PYB,
                'freq': 240,
                'aggregate_phy_steps': 1,
                'gui': False,
                'record': False,
                'obs': ObservationType.KIN,
                'act': ActionType.RPM
            }

        for k, _ in self.env_kwargs.items():
            if k in cfg:
                self.env_kwargs[k] = cfg[k]

        if "print_debug_info" in cfg:
            self.print_debug_info = cfg["print_debug_info"]
        else:
            self.print_debug_info = False

        if "output_folder" in cfg:
            self.output_folder = cfg["output_folder"]
        else:
            self.output_folder = "./results"

        if "plot_observation" in cfg:
            self.plot_observation = cfg["plot_observation"]
        else:
            self.plot_observation = False

        self._cfg = cfg
        self._env_id = cfg.env_id
        self._init_flag = False
        self._replay_path = None

        self._observation_space = gym_pybullet_drones_env_info[cfg.env_id]["observation_space"]
        self._action_space = gym_pybullet_drones_env_info[cfg.env_id]["action_space"]
        self._reward_space = gym_pybullet_drones_env_info[cfg.env_id]["reward_space"]

        self.env_step_count = 0

    def reset(self) -> np.ndarray:
        if not self._init_flag:

            self._env = gym.make(self._env_id, **self.env_kwargs)

            if self.plot_observation:
                self.observation_logger = Logger(
                    logging_freq_hz=int(self._env.SIM_FREQ / self._env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=self.output_folder
                )

            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self.env_step_count = 0
        if self.plot_observation:
            self.observation_logger.log(
                drone=0,
                timestamp=self.env_step_count / self._env.SIM_FREQ,
                state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],
                                 np.resize(np.zeros(4), (4))]),
                control=np.zeros(12)
            )
        if self.print_debug_info:
            if self.env_step_count % self._env.SIM_FREQ == 0:
                self._env.render()
        self.env_step_count += 1

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
        if self.plot_observation:
            self.observation_logger.log(
                drone=0,
                timestamp=self.env_step_count / self._env.SIM_FREQ,
                state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],
                                 np.resize(action, (4))]),
                control=np.zeros(12)
            )

        if self.print_debug_info:
            if self.env_step_count % self._env.SIM_FREQ == 0:
                self._env.render()
        self.env_step_count += 1
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
            if self.print_debug_info:
                self.plot_observation_curve()

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

    def plot_observation_curve(self) -> None:
        if self.plot_observation:
            self.observation_logger.plot()

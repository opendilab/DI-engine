from typing import Optional, Callable
import numpy as np
import copy
import gym
from gym.spaces import Box
import gym_pybullet_drones
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

from easydict import EasyDict


def gym_pybullet_drones_observation_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Callable:
    lower_bound = np.repeat(minimum, dim).astype(dtype)
    upper_bound = np.repeat(maximum, dim).astype(dtype)
    lower_bound[2] = 0.0
    return Box(lower_bound, upper_bound, dtype=dtype)


def drones_action_dim(type_of_action) -> int:
    if type_of_action in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        return 4
    elif type_of_action == ActionType.PID:
        return 3
    elif type_of_action == ActionType.TUN:
        return 6
    elif type_of_action in [ActionType.ONE_D_DYN, ActionType.ONE_D_PID, ActionType.ONE_D_RPM]:
        return 1
    else:
        raise ValueError('Invalid action type.')


def gym_pybullet_drones_action_space(drone_num=1, minimum=-1, maximum=1, dtype=np.float32) -> Callable:

    def _gym_pybullet_drones_action_space(type_of_action) -> Box:
        dim = drones_action_dim(type_of_action)
        return Box(
            np.repeat(minimum, dim * drone_num).astype(dtype),
            np.repeat(maximum, dim * drone_num).astype(dtype),
            dtype=dtype
        )

    return _gym_pybullet_drones_action_space


def gym_pybullet_drones_reward_space(minimum=-10000, maximum=0, dtype=np.float32) -> Callable:
    return Box(np.repeat(minimum, 1).astype(dtype), np.repeat(maximum, 1).astype(dtype), dtype=dtype)


gym_pybullet_drones_env_info = {
    "takeoff-aviary-v0": {
        "observation_space": gym_pybullet_drones_observation_space(12, minimum=-1, maximum=1),
        "action_space": gym_pybullet_drones_action_space(drone_num=1, minimum=-1, maximum=1),
        "reward_space": gym_pybullet_drones_reward_space()
    },
    "flythrugate-aviary-v0": {
        "observation_space": gym_pybullet_drones_observation_space(12, minimum=-1, maximum=1),
        "action_space": gym_pybullet_drones_action_space(drone_num=1, minimum=-1, maximum=1),
        "reward_space": gym_pybullet_drones_reward_space()
    },
}

action_type = {
    "PID": ActionType.PID,
    "DYN": ActionType.DYN,
    "VEL": ActionType.VEL,
    "RPM": ActionType.RPM,
    "TUN": ActionType.TUN,
    "ONE_D_DYN": ActionType.ONE_D_DYN,
    "ONE_D_PID": ActionType.ONE_D_PID,
    "ONE_D_RPM": ActionType.ONE_D_RPM,
}


@ENV_REGISTRY.register('gym_pybullet_drones')
class GymPybulletDronesEnv(BaseEnv):
    """
    Gym_Pybullet_Drones Environment for training and simulating UAV drones in pybullet physical engine.
    The tasks are registered in the standard of gym library.
    url: 'https://github.com/utiasDSL/gym-pybullet-drones'
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = {
        'num_drones': 1,
        'print_debug_info': False,
        'output_folder': "./results",
        'plot_observation': False,
        'freq': 240,
        'aggregate_phy_steps': 1,
        'gui': False,
        'record': False,
        "action_type": "RPM",
    }

    def __init__(self, cfg: dict = {}) -> None:
        self.raw_cfg = copy.deepcopy(cfg)
        for k, v in self.default_config().items():
            if k not in cfg:
                cfg[k] = v

        if cfg["num_drones"] == 1:
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
            # TODO(zjow): develop envs that support multi drones.
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

        self._cfg = cfg

        for k, _ in self.env_kwargs.items():
            if k in cfg:
                self.env_kwargs[k] = cfg[k]

        self.env_kwargs["act"] = action_type[cfg["action_type"]]
        self.action_type = self.env_kwargs["act"]

        self._env_id = cfg.env_id
        self._init_flag = False
        self._replay_path = None

        self._observation_space = gym_pybullet_drones_env_info[cfg.env_id]["observation_space"]
        self._action_space = gym_pybullet_drones_env_info[cfg.env_id]["action_space"](self.action_type)
        self._action_dim = drones_action_dim(self.action_type) * self._cfg["num_drones"]
        self._reward_space = gym_pybullet_drones_env_info[cfg.env_id]["reward_space"]

        self.env_step_count = 0

    def reset(self) -> np.ndarray:
        if not self._init_flag:

            self._env = gym.make(self._env_id, **self.env_kwargs)

            if self._cfg["plot_observation"]:
                self.observation_logger = Logger(
                    logging_freq_hz=int(self._env.SIM_FREQ / self._env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=self._cfg["output_folder"]
                )

            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

        self._eval_episode_return = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self.env_step_count = 0
        if self._cfg["plot_observation"]:
            self.observation_logger.log(
                drone=0,
                timestamp=self.env_step_count / self._env.SIM_FREQ,
                state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],
                                 np.resize(np.zeros(self._action_dim), (4))]),
                control=np.zeros(12)
            )
        if self._cfg["print_debug_info"]:
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
        # action = action.astype('float32')
        obs, rew, done, info = self._env.step(action)
        if self._cfg["plot_observation"]:
            self.observation_logger.log(
                drone=0,
                timestamp=self.env_step_count / self._env.SIM_FREQ,
                state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],
                                 np.resize(action, (4))]),
                control=np.zeros(12)
            )

        if self._cfg["print_debug_info"]:
            if self.env_step_count % self._env.SIM_FREQ == 0:
                self._env.render()
        self.env_step_count += 1
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._cfg["print_debug_info"]:
                self.plot_observation_curve()

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)  # wrapped to be transfered to a array with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        return self.action_space.sample().astype(np.float32)

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
        if self._cfg["plot_observation"]:
            self.observation_logger.plot()

    def clone(self, caller: str) -> 'GymPybulletDronesEnv':
        return GymPybulletDronesEnv(self.raw_cfg)

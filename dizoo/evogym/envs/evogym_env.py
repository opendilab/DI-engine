from typing import Any, Union, List, Optional
import os
import time
import copy
import numpy as np
import gym
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep, EvalEpisodeReturnWrapper
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY

import evogym.envs
from evogym import WorldObject, sample_robot
from evogym.sim import EvoSim


@ENV_REGISTRY.register('evogym')
class EvoGymEnv(BaseEnv):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        env_id='Walker-v0',
        robot='speed_bot',  # refer to 'world data' for more robots configurations
        robot_h=5,  # only used for random robots
        robot_w=5,  # only used for random robots
        robot_pd=None,  # only used for random robots, probability distributions of randomly generated components)
        robot_dir=""  # only used for defined robots, path to the robot config, env/world_data/my_bot.json
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        if 'robot_dir' not in self._cfg.keys():
            self._cfg = '../'

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env()
            self._env.observation_space.dtype = np.float32  # To unify the format of envs in DI-engine
            self._observation_space = self._env.observation_space
            self.num_actuators = self._env.get_actuator_indices('robot').size
            # by default actions space is double (float64), create a new space with type of type float (float32)
            self._action_space = gym.spaces.Box(low=0.6, high=1.6, shape=(self.num_actuators, ), dtype=np.float32)
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            gym.logger.set_level(gym.logger.DEBUG)
            # make render mode compatible with gym
            if gym.version.VERSION > '0.22.0':
                self._env.metadata.update({'render_modes': ["rgb_array"]})
            else:
                self._env.metadata.update({'render.modes': ["rgb_array"]})
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix='rl-video-{}-{}'.format(id(self), time.time())
            )
        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action).astype(np.float32)
        obs, rew, done, info = self._env.step(action)
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def _make_env(self):
        # robot configuration can be read from file or created randomly
        if self._cfg.robot in [None, 'random']:
            h, w = 5, 5
            pd = None
            if 'robot_h' in self._cfg.keys():
                assert self._cfg.robot_h > 0
                h = self._cfg.robot_h
            if 'robot_w' in self._cfg.keys():
                assert self._cfg.robot_w > 0
                w = self._cfg.robot_w
            if 'robot_pd' in self._cfg.keys():
                assert isinstance(self._cfg.robot_pd, np.ndarray)
                assert self._cfg.robot_w > 0
                pd = self._cfg.robot_pd
            structure = sample_robot((h, w), pd)
        else:
            structure = self.read_robot_from_file(self._cfg.robot, self._cfg.robot_dir)
        env = gym.make(self._cfg.env_id, body=structure[0])
        env = EvalEpisodeReturnWrapper(env)
        return env

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        return self.action_space.sample()

    def __repr__(self) -> str:
        return "DI-engine EvoGym Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def read_robot_from_file(file_name, root_dir='../'):
        possible_paths = [
            os.path.join(file_name),
            os.path.join(f'{file_name}.npz'),
            os.path.join(f'{file_name}.json'),
            os.path.join(root_dir, 'world_data', file_name),
            os.path.join(root_dir, 'world_data', f'{file_name}.npz'),
            os.path.join(root_dir, 'world_data', f'{file_name}.json'),
        ]

        best_path = None
        for path in possible_paths:
            if os.path.exists(path):
                best_path = path
                break

        if best_path.endswith('json'):
            robot_object = WorldObject.from_json(best_path)
            return (robot_object.get_structure(), robot_object.get_connections())
        if best_path.endswith('npz'):
            structure_data = np.load(best_path)
            structure = []
            for key, value in structure_data.items():
                structure.append(value)
            return tuple(structure)
        return None

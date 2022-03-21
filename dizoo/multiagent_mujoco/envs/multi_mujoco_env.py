from typing import Any, Union, List
import copy
import numpy as np
import gym
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, update_shape
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray, to_list
from .mujoco_multi import MujocoMulti
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('mujoco_multi')
class MujocoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._cfg.seed = self._seed + np_seed
        elif hasattr(self, '_seed'):
            self._cfg.seed = self._seed
        if not self._init_flag:
            self._env = MujocoMulti(env_args=self._cfg)
            self._init_flag = True
        obs = self._env.reset()
        # print(obs)
        # obs['agent_state'] = to_ndarray(obs['agent_state']).astype('float32')
        # obs['global_state'] = to_ndarray(obs['global_state']).astype('float32')
        self._final_eval_reward = 0.
        self.env_info = self._env.get_env_info()
        self._observation_space = gym.spaces.Box(
            low=np.float32("inf"), high=np.float32("-inf"), shape={'agent_state': (self.env_info['obs_shape'],),
                                                                   'global_state': (self.env_info['state_shape'],), },
            dtype=np.float32
        )
        self._action_space = self._env.action_space
        self._reward_space = gym.spaces.Box(
            low=np.float32("-inf"), high=np.float32("inf"), shape=(1,), dtype=np.float32
        )

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
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        # obs = to_ndarray(obs).astype('float32')
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    # def info(self) -> BaseEnvInfo:
    #     env_info = self._env.get_env_info()
    #     info = BaseEnvInfo(
    #         agent_num=env_info['n_agents'],
    #         obs_space=EnvElementInfo(
    #             shape={
    #                 'agent_state': env_info['obs_shape'],
    #                 'global_state': env_info['state_shape'],
    #             },
    #             value={
    #                 'min': np.float32("-inf"),
    #                 'max': np.float32("inf"),
    #                 'dtype': np.float32
    #             },
    #         ),
    #         act_space=EnvElementInfo(
    #             shape=env_info['action_spaces'],
    #             value={
    #                 'min': np.float32("-inf"),
    #                 'max': np.float32("inf"),
    #                 'dtype': np.float32
    #             },
    #         ),
    #         rew_space=EnvElementInfo(
    #             shape=1,
    #             value={
    #                 'min': np.float64("-inf"),
    #                 'max': np.float64("inf")
    #             },
    #         ),
    #         use_wrappers=None,
    #     ),
    #     return info

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
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

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "DI-engine Multi-agent Mujoco Env({})".format(self._cfg.env_id)

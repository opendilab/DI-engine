from typing import Any, List, Union, Optional
import time
import copy
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY

import bsuite
from bsuite.utils import gym_wrapper
from bsuite import sweep

BSUITE_INFO_DICT = {
    'memory_len': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            (1, 3),
            {
                'min': 0.,
                'max': 1.,
                'dtype': np.float32,
            },
        ),
        act_space=EnvElementInfo(
            (1, ),
            {
                'min': 0,
                'max': 1,
                'dtype': int,
            },
        ),
        rew_space=EnvElementInfo(
            (1, ),
            {
                'min': -1.,
                'max': 1.,
                'dtype': np.float64,
            },
        ),
        use_wrappers=None,
    )
}


@ENV_REGISTRY.register('bsuite')
class BSuiteEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self.env_id = cfg.env_id
        self.env_name = self.env_id.split('/')[0]

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            raw_env = bsuite.load_from_id(bsuite_id=self.env_id)
            self._env = gym_wrapper.GymFromDMEnv(raw_env)
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

    def step(self, action: np.int32) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        settings_info = sweep.SETTINGS[self.env_id]  # additional info that are specific to each env configuration
        if self.env_name in BSUITE_INFO_DICT:
            info = copy.deepcopy(BSUITE_INFO_DICT[self.env_name])
            for k, v in settings_info.items():
                info[k] = v
            info['num_episodes'] = self._env.bsuite_num_episodes
            return info
        else:
            raise NotImplementedError('{} not found in BSUITE_INFO_DICT [{}]'\
                .format(self.env_name, BSUITE_INFO_DICT.keys()))

    def __repr__(self) -> str:
        return "DI-engine BSuite Env({})".format(self.env_id)

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

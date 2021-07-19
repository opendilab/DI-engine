from typing import Any, List, Union, Optional
from collections import namedtuple
import time
import gym
import numpy as np
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY

MINIGRID_INFO_DICT = {
    'MiniGrid-Empty-8x8-v0': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(shape=(2739, ), value={
            'min': 0,
            'max': 5,
            'dtype': np.float32
        }),
        act_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 7,
            'dtype': np.int64,
        }),
        rew_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 1,
            'dtype': np.float32
        }),
        use_wrappers=None,
    ),
}


@ENV_REGISTRY.register('minigrid')
class MiniGridEnv(BaseEnv):
    config = dict(
        env_id='MiniGrid-Empty-8x8-v0',
        flat_obs=True,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._env_id = cfg.env_id
        self._flat_obs = cfg.flat_obs

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id)
            if self._flat_obs:
                self._env = FlatObsWrapper(self._env)
                # self._env = RGBImgPartialObsWrapper(self._env)
                # self._env = ImgObsWrapper(self._env)
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

    def render(self) -> None:
        self._env.render()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.squeeze()  # 0-dim tensor
        obs, rew, done, info = self._env.step(action)
        rew = float(rew)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        return MINIGRID_INFO_DICT[self._env_id]

    def __repr__(self) -> str:
        return "DI-engine MiniGrid Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        raise NotImplementedError

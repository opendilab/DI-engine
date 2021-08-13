from namedlist import namedlist
import numpy as np
import gym
from typing import Any, Union, List, Optional
import copy
import slimevolleygym
from typing import Optional

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray


@ENV_REGISTRY.register('slime_volley')
class SlimeVolleyEnv(BaseEnv):

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def step(self, action1: np.ndarray, action2: Optional[np.ndarray] = None):
        assert isinstance(action1, np.ndarray), type(action1)
        assert action2 is None or isinstance(action1, np.ndarray), type(action2)
        if action1.shape == (1, ):
            action1 = action1.squeeze()  # 0-dim tensor
        if action2 is not None and action2.shape == (1, ):
            action2 = action2.squeeze()  # 0-dim tensor
        obs, rew, done, info = self._env.step(action1, action2)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def reset(self):
        if not self._init_flag:
            self._env = gym.make(self._cfg.env_id)
            if self._replay_path is not None:
                self._env = gym.wrappers.Monitor(
                    self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
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
        return obs

    def info(self):
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=2,
            obs_space=T(
                (12, ),
                {
                    'min': [float("-inf") for _ in range(12)],
                    'max': [float("inf") for _ in range(12)],
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            act_space=T(
                (3, ),
                {
                    'min': 0,
                    'max': 2,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': float("-inf"),
                    'max': float("-inf"),
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self):
        return "DI-engine Slam Volley Env"
    
    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

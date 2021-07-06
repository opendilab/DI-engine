from typing import Any, List, Union, Optional
import time
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, FrameStack
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.envs.common.common_function import affine_transform
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('bipedalwalker')
class BipedalWalkerEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._act_scale = cfg.act_scale
        self._rew_clip = cfg.rew_clip
        self._replay_path = cfg.replay_path

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make('BipedalWalker-v3')
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.Monitor(
                self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
            )
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
        if self._act_scale:
            action_range = self.info().act_space.value
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])

        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if self._rew_clip:
            rew = max(-10, rew)
        rew = np.float32(rew)

        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (24, ),
                {
                    'min': [float("-inf")] * 24,
                    'max': [float("inf")] * 24,
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            act_space=T(
                (4, ),
                {
                    'min': -1.0,
                    'max': 1.0,
                    'dtype': np.float32,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': -300,
                    'max': 400,
                    'dtype': np.float32,
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "DI-engine BipedalWalker Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        # this function can lead to the meaningless result
        self._env = gym.wrappers.Monitor(
            self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
        )

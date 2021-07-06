from typing import Any, List, Union, Optional
import time
import gym
import torch
import copy
import numpy as np
from easydict import EasyDict
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY


def disable_gym_view_window():
    from gym.envs.classic_control import rendering
    import pyglet

    def get_window(width, height, display):
        screen = display.get_screens()
        config = screen[0].get_best_config()
        context = config.create_context(None)
        return pyglet.window.Window(
            width=width, height=height, display=display, config=config, context=context, visible=False
        )

    rendering.get_window = get_window


@ENV_REGISTRY.register('cartpole')
class CartPoleEnv(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None

    def reset(self) -> torch.Tensor:
        if not self._init_flag:
            self._env = gym.make('CartPole-v0')
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

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.squeeze()  # 0-dim tensor
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
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
                (4, ),
                {
                    'min': [-4.8, float("-inf"), -0.42, float("-inf")],
                    'max': [4.8, float("inf"), 0.42, float("inf")],
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            act_space=T(
                (1, ),
                {
                    'min': 0,
                    'max': 2,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': 0.0,
                    'max': 1.0
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "DI-engine CartPole Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

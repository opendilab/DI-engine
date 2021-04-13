from typing import Any, List, Union, Optional
import time
import gym
import torch
import numpy as np
from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from nervex.utils import ENV_REGISTRY


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

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False

    def reset(self) -> torch.Tensor:
        if not self._init_flag:
            self._env = gym.make('CartPole-v0')
            self._init_flag = True
        if hasattr(self, '_seed'):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs)
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.squeeze()  # 0-dim tensor
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (4, ), {
                    'min': [-4.8, float("-inf"), -0.42, float("-inf")],
                    'max': [4.8, float("inf"), 0.42, float("inf")],
                    'dtype': float,
                }, None, None
            ),
            # [min, max)
            act_space=T((2, ), {
                'min': 0,
                'max': 2
            }, None, None),
            rew_space=T((1, ), {
                'min': 0.0,
                'max': 1.0
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX CartPole Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        # this function can lead to the meaningless result
        # disable_gym_view_window()
        self._env = gym.wrappers.Monitor(
            self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
        )

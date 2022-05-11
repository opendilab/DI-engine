from typing import Any, List, Union, Sequence
import copy
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray, to_list
from .atari_wrappers import wrap_deepmind

from pprint import pprint


def PomdpEnv(cfg, only_info=False):
    '''
    For debug purpose, create an env follow openai gym standard so it can be widely test by
    other library with same environment setting in DI-engine
    env = PomdpEnv(cfg)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    '''
    env = wrap_deepmind(
        cfg.env_id,
        frame_stack=cfg.frame_stack,
        episode_life=cfg.is_train,
        clip_rewards=cfg.is_train,
        warp_frame=cfg.warp_frame,
        use_ram=cfg.use_ram,
        render=cfg.render,
        pomdp=cfg.pomdp,
        only_info=only_info,
    )
    return env


@ENV_REGISTRY.register('pomdp')
class PomdpAtariEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False

    def reset(self) -> Sequence:
        if not self._init_flag:
            self._env = self._make_env(only_info=False)
            self._init_flag = True
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
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
        action = action.item()
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def _make_env(self, only_info=False):
        return wrap_deepmind(
            self._cfg.env_id,
            episode_life=self._cfg.is_train,
            clip_rewards=self._cfg.is_train,
            pomdp=self._cfg.pomdp,
            frame_stack=self._cfg.frame_stack,
            warp_frame=self._cfg.warp_frame,
            use_ram=self._cfg.use_ram,
            only_info=only_info,
        )

    def __repr__(self) -> str:
        return "DI-engine POMDP Atari Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

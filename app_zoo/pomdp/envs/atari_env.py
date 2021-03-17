from typing import Any, List, Union, Sequence
import copy
import torch
import gym
import numpy as np
from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.utils import ENV_REGISTRY
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from .atari_wrappers import wrap_deepmind

from pprint import pprint


def PomdpEnv(cfg):
    '''
    For debug purpose, create an env follow openai gym standard so it can be widely test by
    other library with same environment setting in nerveX
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
    )
    return env


@ENV_REGISTRY.register('pomdp')
class PomdpAtariEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env = wrap_deepmind(
            cfg.env_id,
            frame_stack=cfg.frame_stack,
            episode_life=cfg.is_train,
            clip_rewards=cfg.is_train,
            warp_frame=cfg.warp_frame,
            use_ram=cfg.use_ram,
            render=cfg.render,
            pomdp=cfg.pomdp,
        )

    def reset(self) -> Sequence:
        if hasattr(self, '_seed'):
            np.random.seed(self._seed)
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        rew_range = self._env.reward_range
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(self._env.observation_space.shape, {'dtype': np.float32}, None, None),
            act_space=T((self._env.action_space.n, ), {'dtype': np.float32}, None, None),
            rew_space=T(1, {
                'min': rew_range[0],
                'max': rew_range[1],
                'dtype': np.float32
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX POMDP Atari Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_actor_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.pop('actor_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

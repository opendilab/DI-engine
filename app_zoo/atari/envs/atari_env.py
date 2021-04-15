from typing import Any, List, Union, Sequence
import copy
import torch
import numpy as np
from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.utils import ENV_REGISTRY
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from .atari_wrappers import wrap_deepmind, wrap_deepmind_mr


ATARIENV_INFO_DICT = {
    'PongNoFrameskip-v4': BaseEnvInfo(
        agent_num=1, 
        obs_space=EnvElementInfo(
            shape=(4, 84, 84), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        act_space=EnvElementInfo(
            shape=(6,), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        rew_space=EnvElementInfo(
            shape=1, 
            value={'min': -1, 'max': 1, 'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        )
    ),
    'QbertNoFrameskip-v4': BaseEnvInfo(
        agent_num=1, 
        obs_space=EnvElementInfo(
            shape=(4, 84, 84), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        act_space=EnvElementInfo(
            shape=(6,), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        rew_space=EnvElementInfo(
            shape=1, 
            value={'min': -1, 'max': 1, 'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        )
    ),
    'SpaceInvadersNoFrameskip-v4': BaseEnvInfo(
        agent_num=1, 
        obs_space=EnvElementInfo(
            shape=(4, 84, 84), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        act_space=EnvElementInfo(
            shape=(6,), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        rew_space=EnvElementInfo(
            shape=1, 
            value={'min': -1, 'max': 1, 'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        )
    ),
    'MontezumaRevengeDeterministic-v4': BaseEnvInfo(
        agent_num=1, 
        obs_space=EnvElementInfo(
            shape=(4, 84, 84), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        act_space=EnvElementInfo(
            shape=(18,), 
            value={'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        ), 
        rew_space=EnvElementInfo(
            shape=1, 
            value={'min': -1, 'max': 1, 'dtype': np.float32}, 
            to_agent_processor=None, 
            from_agent_processor=None
        )
    ),
}


@ENV_REGISTRY.register("atari")
class AtariEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = wrap_deepmind(
                self._cfg.env_id, frame_stack=self._cfg.frame_stack, episode_life=self._cfg.is_train, 
                clip_rewards=self._cfg.is_train
            )
            self._init_flag = True
        if hasattr(self, '_seed'):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed
        np.random.seed(self._seed)

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
        if self._cfg.env_id in ATARIENV_INFO_DICT:
            return ATARIENV_INFO_DICT[self._cfg.env_id]
        else:
            raise NotImplementedError('{} not found in ATARIENV_INFO_DICT [{}]'\
                .format(self._cfg.env_id, ATARIENV_INFO_DICT.keys()))

    def __repr__(self) -> str:
        return "nerveX Atari Env({})".format(self._cfg.env_id)

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


@ENV_REGISTRY.register('atari_mr')
class AtariEnvMR(AtariEnv):
        
    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = wrap_deepmind_mr(
                self._cfg.env_id, frame_stack=self._cfg.frame_stack, episode_life=self._cfg.is_train, 
                clip_rewards=self._cfg.is_train
            )
            self._init_flag = True
        if hasattr(self, '_seed'):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        return obs

    def info(self) -> BaseEnvInfo:
        if self._cfg.env_id in ATARIENV_INFO_DICT:
            return ATARIENV_INFO_DICT[self._cfg.env_id]
        else:
            raise NotImplementedError('{} not found in ATARIENV_INFO_DICT [{}]'\
                .format(self._cfg.env_id, ATARIENV_INFO_DICT.keys()))

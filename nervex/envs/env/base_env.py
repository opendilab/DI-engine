from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import logging
import gym
import copy
import numpy as np
from namedlist import namedlist
from collections import namedtuple
from nervex.utils import import_module, ENV_REGISTRY
from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list

BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])
BaseEnvInfo = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])


class BaseEnv(gym.Env):
    """
    Overview: basic environment class, extended from `gym.Env`
    Interface: __init__
    Property: timestep
    """

    @abstractmethod
    def __init__(self, cfg: dict) -> None:
        """
        Overview: lazy init, only parameters will be initialized in `self.__init__()`
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        """
        Overview: Resets the env to an initial state and returns an initial observation.
                  ABS Method from `gym.Env`.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Overview: Environments will automatically close() themselves when garbage collected or exits.
                  ABS Method from `gym.Env`.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> 'BaseEnv.timestep':
        """
        Overview: Run one timestep of the environment's dynamics.
                  ABS Method from `gym.Env`.
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        """
        Overview: Sets the seed for this env's random number generator(s).
                  ABS Method from `gym.Env`.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> 'BaseEnv.info':
        """
        Overview: Show space in code and return namedlist.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @staticmethod
    def create_actor_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.get('actor_env_num', 1)
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.get('evaluator_env_num', 1)
        return [cfg for _ in range(evaluator_env_num)]

    # optional method
    def enable_save_replay(self, replay_path: str) -> None:
        raise NotImplementedError


class NervexEnvWrapper(BaseEnv):

    def __init__(self, env: gym.Env, cfg: dict = None) -> None:
        self._cfg = cfg
        self._env = env

    # override
    def reset(self) -> None:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        return obs

    # override
    def close(self) -> None:
        self._env.close()

    # override
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    # override
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
        obs_space = self._env.observation_space
        act_space = self._env.action_space
        return BaseEnvInfo(
            agent_num=1, 
            obs_space=EnvElementInfo(
                shape=obs_space.shape, 
                value={'min': obs_space.low, 'max': obs_space.high, 'dtype': np.float32}, 
                to_agent_processor=None, 
                from_agent_processor=None
            ), 
            act_space=EnvElementInfo(
                shape=(act_space.n, ), 
                value={'min': 0, 'max': act_space.n, 'dtype': np.float32}, 
                to_agent_processor=None, 
                from_agent_processor=None
            ), 
            rew_space=EnvElementInfo(
                shape=1, 
                value={'min': -1, 'max': 1, 'dtype': np.float32}, 
                to_agent_processor=None, 
                from_agent_processor=None
            )
        )

    def __repr__(self) -> str:
        return "nerveX Env({})".format(self._cfg.env_id)

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


def get_vec_env_setting(cfg: dict) -> Tuple[type, List[dict], List[dict]]:
    import_module(cfg.get('import_names', []))
    env_fn = ENV_REGISTRY[cfg.env_type]
    actor_env_cfg = env_fn.create_actor_env_cfg(cfg)
    evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)
    return env_fn, actor_env_cfg, evaluator_env_cfg

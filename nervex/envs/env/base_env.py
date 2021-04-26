from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import logging
import gym
import copy
import numpy as np
from namedlist import namedlist
from collections import namedtuple
from nervex.utils import import_module, ENV_REGISTRY

BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])
BaseEnvInfo = namedlist('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space', 'use_wrappers'])


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
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [cfg for _ in range(evaluator_env_num)]

    # optional method
    def enable_save_replay(self, replay_path: str) -> None:
        raise NotImplementedError


def get_vec_env_setting(cfg: dict) -> Tuple[type, List[dict], List[dict]]:
    import_module(cfg.get('import_names', []))
    env_fn = ENV_REGISTRY[cfg.env_type]
    collector_env_cfg = env_fn.create_collector_env_cfg(cfg)
    evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)
    return env_fn, collector_env_cfg, evaluator_env_cfg

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, List, Tuple
import logging
from nervex.utils import import_module, ENV_REGISTRY

BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])
BaseEnvInfo = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])


class BaseEnv(ABC):
    """
    Overview: basic environment class
    Interface: __init__
    Property: timestep
    """

    @abstractmethod
    def __init__(self, cfg: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> 'BaseEnv.timestep':
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> 'BaseEnv.info':
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.get('collector_env_num', 1)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.get('evaluator_env_num', 1)
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

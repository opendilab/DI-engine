from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, List, Tuple
import warnings
from nervex.utils import import_module

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
    def create_actor_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.pop('actor_env_num', 1)
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        return [cfg for _ in range(evaluator_env_num)]

    # optional method
    def enable_save_replay(self, replay_path: str) -> None:
        raise NotImplementedError


env_mapping = {}


def register_env(name: str, env: type) -> None:
    assert issubclass(env, BaseEnv)
    if name in env_mapping:
        warnings.warn("env name {} has already been registered".format(name))
    env_mapping[name] = env


def get_vec_env_setting(cfg: dict) -> Tuple[type, List[dict], List[dict]]:
    import_module(cfg.pop('import_names', []))
    if cfg.env_type in env_mapping:
        env_fn = env_mapping[cfg.env_type]
    else:
        raise KeyError("invalid env type: {}".format(cfg.env_type))
    actor_env_cfg = env_fn.create_actor_env_cfg(cfg)
    evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)
    return env_fn, actor_env_cfg, evaluator_env_cfg

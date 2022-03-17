from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import gym
import copy
from easydict import EasyDict
from namedlist import namedlist
from collections import namedtuple
from ding.utils import import_module, ENV_REGISTRY

BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])
BaseEnvInfo = namedlist('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space', 'use_wrappers'])


class BaseEnv(ABC, gym.Env):
    """
    Overview:
        basic environment class, extended from ``gym.Env``
    Interface:
        ``__init__``, ``reset``, ``close``, ``step``, ``create_collector_env_cfg``, \
        ``create_evaluator_env_cfg``, ``enable_save_replay``
    """

    @abstractmethod
    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Lazy init, only parameters will be initialized in ``self.__init__()``
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        """
        Overview:
            Resets the env to an initial state and returns an initial observation. Abstract Method from ``gym.Env``.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Overview:
            Environments will automatically ``close()`` themselves when garbage collected or exits. \
                Abstract Method from ``gym.Env``.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> 'BaseEnv.timestep':
        """
        Overview:
            Run one timestep of the environment's dynamics. Abstract Method from ``gym.Env``.
        Arguments:
            - action (:obj:`Any`): the ``action`` input to step with
        Returns:
            - timestep (:obj:`BaseEnv.timestep`)
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        """
        Overview:
            Sets the seed for this env's random number generator(s). Abstract Method from ``gym.Env``.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config.
        Arguments:
            - cfg (:obj:`Dict`) Env config, same config where ``self.__init__()`` takes arguments from
        Returns:
            - List of ``cfg`` including all of the collector env's config
        """
        collector_env_num = cfg.pop('collector_env_num')
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config.
        Arguments:
            - cfg (:obj:`Dict`) Env config, same config where ``self.__init__()`` takes arguments from
        Returns:
            - List of ``cfg`` including all of the evaluator env's config
        """
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [cfg for _ in range(evaluator_env_num)]

    # optional method
    def enable_save_replay(self, replay_path: str) -> None:
        """
        Overview:
            Save replay file in the given path, need to be self-implemented.
        Arguments:
            - replay_path(:obj:`str`): Storage path.
        """
        raise NotImplementedError


def get_vec_env_setting(cfg: dict) -> Tuple[type, List[dict], List[dict]]:
    """
    Overview:
       Get vectorized env setting(env_fn, collector_env_cfg, evaluator_env_cfg)
    Arguments:
        - cfg (:obj:`Dict`) Env config, same config where ``self.__init__()`` takes arguments from
    Returns:
        - env_fn (:obj:`type`): Callable object, call it with proper arguments and then get a new env instance.
        - collector_env_cfg (:obj:`List[dict]`): A list contains the config of collecting data envs.
        - evaluator_env_cfg (:obj:`List[dict]`): A list contains the config of evaluation envs.

    .. note::
        elements(env config) in collector_env_cfg/evaluator_env_cfg can be different, such as server ip and port.

    """
    import_module(cfg.get('import_names', []))
    env_fn = ENV_REGISTRY.get(cfg.type)
    collector_env_cfg = env_fn.create_collector_env_cfg(cfg)
    evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)
    return env_fn, collector_env_cfg, evaluator_env_cfg


def get_env_cls(cfg: EasyDict) -> type:
    """
    Overview:
       Get the env class by correspondng module of ``cfg`` and return the callable class
    Arguments:
        - cfg (:obj:`Dict`) Env config, same config where ``self.__init__()`` takes arguments from
    Returns:
        - Env module as the corresponding callable class

    """
    import_module(cfg.get('import_names', []))
    return ENV_REGISTRY.get(cfg.type)


def create_model_env(cfg: EasyDict) -> Any:
    cfg = copy.deepcopy(cfg)
    model_env_fn = get_env_cls(cfg)
    cfg.pop('import_names')
    cfg.pop('type')
    return model_env_fn(**cfg)

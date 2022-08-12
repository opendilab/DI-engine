from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import gym
import copy
from easydict import EasyDict
from collections import namedtuple
from ding.utils import import_module, ENV_REGISTRY

BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])


# for solving multiple inheritance metaclass conflict between gym and ABC
class FinalMeta(type(gym.Env), type(ABC)):
    pass


class BaseEnv(gym.Env, ABC, metaclass=FinalMeta):
    """
    Overview:
        Basic environment class, extended from ``gym.Env``
    Interface:
        ``__init__``, ``reset``, ``close``, ``step``, ``random_action``, ``create_collector_env_cfg``, \
        ``create_evaluator_env_cfg``, ``enable_save_replay``
    """

    @abstractmethod
    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Lazy init, only related arguments will be initialized in ``__init__`` method, and the concrete \
            env will be initialized the first time ``reset`` method is called.
        Arguments:
            - cfg (:obj:`dict`): Environment configuration in dict type.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        """
        Overview:
            Reset the env to an initial state and returns an initial observation.
        Returns:
            - obs (:obj:`Any`): Initial observation after reset.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Overview:
            Close env and all the related resources, it should be called after the usage of env instance.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> 'BaseEnv.timestep':
        """
        Overview:
            Run one timestep of the environment's dynamics/simulation.
        Arguments:
            - action (:obj:`Any`): The ``action`` input to step with.
        Returns:
            - timestep (:obj:`BaseEnv.timestep`): The result timestep of env executing one step.
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        """
        Overview:
            Set the seed for this env's random number generator(s).
        Arguments:
            - seed (:obj:`Any`): Random seed.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Overview:
            Return the information string of this env instance.
        Returns:
            - info (:obj:`str`): Information of this env instance, like type and arguments.
        """
        raise NotImplementedError

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in env manager \
            (a series of vectorized env), and this method is mainly responsible for envs collecting data.
        Arguments:
            - cfg (:obj:`dict`): Original input env config, which needs to be transformed into the type of creating \
                env instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config collector envs.

        .. note::
            Elements(env config) in collector_env_cfg/evaluator_env_cfg can be different, such as server ip and port.
        """
        collector_env_num = cfg.pop('collector_env_num')
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in env manager \
            (a series of vectorized env), and this method is mainly responsible for envs evaluating performance.
        Arguments:
            - cfg (:obj:`dict`): Original input env config, which needs to be transformed into the type of creating \
                env instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config evaluator envs.
        """
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [cfg for _ in range(evaluator_env_num)]

    # optional method
    def enable_save_replay(self, replay_path: str) -> None:
        """
        Overview:
            Save replay file in the given path, and this method need to be self-implemented by each env class.
        Arguments:
            - replay_path (:obj:`str`): The path to save replay file.
        """
        raise NotImplementedError

    # optional method
    def random_action(self) -> Any:
        """
        Overview:
            Return random action generated from the original action space, usually it is convenient for test.
        Returns:
            - random_action (:obj:`Any`): Action generated randomly.
        """
        pass


def get_vec_env_setting(cfg: dict, collect: bool = True, eval_: bool = True) -> Tuple[type, List[dict], List[dict]]:
    """
    Overview:
        Get vectorized env setting (env_fn, collector_env_cfg, evaluator_env_cfg).
    Arguments:
        - cfg (:obj:`dict`): Original input env config in user config, such as ``cfg.env``.
    Returns:
        - env_fn (:obj:`type`): Callable object, call it with proper arguments and then get a new env instance.
        - collector_env_cfg (:obj:`List[dict]`): A list contains the config of collecting data envs.
        - evaluator_env_cfg (:obj:`List[dict]`): A list contains the config of evaluation envs.

    .. note::
        Elements (env config) in collector_env_cfg/evaluator_env_cfg can be different, such as server ip and port.

    """
    import_module(cfg.get('import_names', []))
    env_fn = ENV_REGISTRY.get(cfg.type)
    collector_env_cfg = env_fn.create_collector_env_cfg(cfg) if collect else None
    evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg) if eval_ else None
    return env_fn, collector_env_cfg, evaluator_env_cfg


def get_env_cls(cfg: EasyDict) -> type:
    """
    Overview:
        Get the env class by correspondng module of ``cfg`` and return the callable class.
    Arguments:
        - cfg (:obj:`dict`): Original input env config in user config, such as ``cfg.env``.
    Returns:
        - env_cls_type (:obj:`type`): Env module as the corresponding callable class type.
    """
    import_module(cfg.get('import_names', []))
    return ENV_REGISTRY.get(cfg.type)


def create_model_env(cfg: EasyDict) -> Any:
    """
    Overview:
        Create model env, which is used in model-based RL.
    """
    cfg = copy.deepcopy(cfg)
    model_env_fn = get_env_cls(cfg)
    cfg.pop('import_names')
    cfg.pop('type')
    return model_env_fn(**cfg)

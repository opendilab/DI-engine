from abc import ABC
from types import MethodType
from typing import Type, Union, Any, List, Callable, Iterable, Dict, Optional
from functools import partial, wraps
from easydict import EasyDict
import copy
from collections import namedtuple
import numbers
import torch
import enum
import time
import traceback
import signal
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from nervex.utils import ENV_MANAGER_REGISTRY, import_module, deep_merge_dicts
from nervex.envs.env.base_env import BaseEnvTimestep
from nervex.utils.time_helper import WatchDog


class EnvState(enum.IntEnum):
    VOID = 0
    INIT = 1
    RUN = 2
    RESET = 3
    DONE = 4
    ERROR = 5


def retry_wrapper(func: Callable = None, max_retry: int = 10, waiting_time: float = 0.1) -> Callable:
    """
    Overview:
        Retry the function until exceeding the maximum retry times.
    """

    if func is None:
        return partial(retry_wrapper, max_retry=max_retry)

    @wraps(func)
    def wrapper(*args, **kwargs):
        exceptions = []
        for _ in range(max_retry):
            try:
                ret = func(*args, **kwargs)
                return ret
            except BaseException as e:
                exceptions.append(e)
                time.sleep(waiting_time)
        e_info = ''.join(
            [
                'Retry {} failed from:\n {}\n'.format(i, ''.join(traceback.format_tb(e.__traceback__)) + str(e))
                for i, e in enumerate(exceptions)
            ]
        )
        func_exception = Exception("Function {} runtime error:\n{}".format(func, e_info))
        raise RuntimeError("Function {} has exceeded max retries({})".format(func, max_retry)) from func_exception

    return wrapper


def timeout_wrapper(func: Callable = None, timeout: int = 10) -> Callable:
    if func is None:
        return partial(timeout_wrapper, timeout=timeout)

    @wraps(func)
    def wrapper(*args, **kwargs):
        watchdog = WatchDog(timeout)
        watchdog.start()
        try:
            ret = func(*args, **kwargs)
            return ret
        except BaseException as e:
            raise e
        finally:
            watchdog.stop()

    return wrapper


@ENV_MANAGER_REGISTRY.register('base')
class BaseEnvManager(object):
    """
    Overview:
        Create a BaseEnvManager to manage multiple environments.
    Interfaces:
        reset, step, seed, close, enable_save_replay, launch, env_info
    Properties:
        env_num, ready_obs, done, method_name_list
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    config = dict(
        episode_num=float("inf"),
        max_retry=1,
        step_timeout=60,
        reset_timeout=60,
        retry_waiting_time=0.1,
    )

    def __init__(
            self,
            env_fn: List[Callable],
            cfg: EasyDict = EasyDict({}),
    ) -> None:
        """
        Overview:
            Initialize the BaseEnvManager.
        Arguments:
            - env_fn (:obj:`List[Callable]`): the function to create environment
        """
        self._cfg = deep_merge_dicts(self.default_config(), cfg)
        self._env_fn = env_fn
        self._env_num = len(self._env_fn)
        self._transform = partial(to_ndarray)
        self._inv_transform = partial(to_tensor, dtype=torch.float32)
        self._closed = True
        self._env_replay_path = None
        # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
        self._env_ref = self._env_fn[0]()
        self._env_states = {i: EnvState.VOID for i in range(self._env_num)}

        self._episode_num = self._cfg.episode_num
        self._max_retry = self._cfg.max_retry
        self._step_timeout = self._cfg.step_timeout
        self._reset_timeout = self._cfg.reset_timeout
        self._retry_waiting_time = self._cfg.retry_waiting_time

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def ready_obs(self) -> Dict[int, Any]:
        """
        Overview:
            Get the next observations(in ``torch.Tensor`` type) and corresponding env id.
        Return:
            A dictionary with observations and their environment IDs.
        Example:
            >>>     obs_dict = env_manager.ready_obs
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
        """
        return self._inv_transform(
            {i: self._ready_obs[i]
             for i in range(self.env_num) if self._env_episode_count[i] < self._episode_num}
        )

    @property
    def done(self) -> bool:
        return all([s == EnvState.DONE for s in self._env_states.values()])

    @property
    def method_name_list(self) -> list:
        return ['reset', 'step', 'seed', 'close', 'enable_save_replay']

    @property
    def active_env(self) -> List[int]:
        return [i for i, s in self._env_states.items() if s == EnvState.RUN]

    def __getattr__(self, key: str) -> Any:
        """
        Note: if a python object doesn't have the attribute named key, it will call this method
        """
        # We suppose that all envs have the same attributes.
        # If you need different envs, please implement other env managers.
        if not hasattr(self._env_ref, key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._env_ref), key))
        if isinstance(getattr(self._env_ref, key), MethodType) and key not in self.method_name_list:
            raise RuntimeError("env getattr doesn't support method({}), please override method_name_list".format(key))
        self._check_closed()
        return [getattr(env, key) if hasattr(env, key) else None for env in self._envs]

    def _check_closed(self):
        """
        Overview:
            Check whether the env manager is closed. Will be called in ``__getattr__`` and ``step``.
        """
        assert not self._closed, "env manager is closed, please use the alive env manager"

    def launch(self, reset_param: Optional[List[dict]] = None) -> None:
        """
        Overview:
            Set up the environments and hyper-params.
        Arguments:
            - reset_param (:obj:`Optional[List[dict]]`): List of reset parameters for each environment.
        """
        assert self._closed, "please first close the env manager"
        self._create_state()
        # set seed
        if hasattr(self, '_env_seed'):
            for env, s in zip(self._envs, self._env_seed):
                if self._env_dynamic_seed is not None:
                    env.seed(s, self._env_dynamic_seed)
                else:
                    env.seed(s)
        self.reset(reset_param)

    def _create_state(self) -> None:
        self._env_episode_count = {i: 0 for i in range(self.env_num)}
        self._ready_obs = {i: None for i in range(self.env_num)}
        self._envs = [e() for e in self._env_fn]
        # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
        self._env_ref = self._envs[0]
        assert len(self._envs) == self._env_num
        self._env_states = {i: EnvState.INIT for i in range(self._env_num)}
        if self._env_replay_path is not None:
            for e, s in zip(self._envs, self._env_replay_path):
                e.enable_save_replay(s)
        self._closed = False

    def reset(self, reset_param: List[dict] = None) -> None:
        """
        Overview:
            Reset the environments and hyper-params.
        Arguments:
            - reset_param (:obj:`List`): list of reset parameters for each environment.
        """
        self._check_closed()
        if reset_param is None:
            reset_param = [{} for _ in range(self.env_num)]
        self._reset_param = reset_param
        for i in range(self.env_num):
            self._reset(i)

    def _reset(self, env_id: int) -> None:

        @retry_wrapper(max_retry=self._max_retry, waiting_time=self._retry_waiting_time)
        @timeout_wrapper(timeout=self._reset_timeout)
        def reset_fn():
            return self._envs[env_id].reset(**self._reset_param[env_id])

        try:
            obs = reset_fn()
        except Exception as e:
            self._env_states[env_id] = EnvState.ERROR
            self.close()
            raise e
        self._ready_obs[env_id] = obs
        self._env_states[env_id] = EnvState.RUN

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        """
        Overview:
            All envs Wrapper of step function in the environment.
        Arguments:
            - actions (:obj:`Dict[int, Any]`): {env_id: action}
        Return:
            - timesteps (:obj:`Dict[int, namedtuple]`): {env_id: timestep}. timestep is in ``torch.Tensor`` type.
        Note:
            - The env_id that appears in ``actions`` will also be returned in ``timesteps``.
            - Once an environment is done, it is reset immediately.
        Example:
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
            >>>     timesteps = env_manager.step(actions_dict):
            >>>     for env_id, timestep in timesteps.items():
            >>>         pass
        """

        self._check_closed()
        timesteps = {}
        for env_id, act in actions.items():
            act = self._transform(act)
            timesteps[env_id] = self._step(env_id, act)
            if timesteps[env_id].info.get('abnormal', False):
                self._env_states[env_id] = EnvState.RESET
                self._reset(env_id)
            elif timesteps[env_id].done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num:
                    self._env_states[env_id] = EnvState.RESET
                    self._reset(env_id)
                else:
                    self._env_states[env_id] = EnvState.DONE
            else:
                self._ready_obs[env_id] = timesteps[env_id].obs
        return self._inv_transform(timesteps)

    def _step(self, env_id: int, act: Any) -> namedtuple:

        @retry_wrapper(max_retry=self._max_retry, waiting_time=self._retry_waiting_time)
        @timeout_wrapper(timeout=self._step_timeout)
        def step_fn():
            return self._envs[env_id].step(act)

        try:
            ret = step_fn()
            return ret
        except Exception as e:
            self._env_states[env_id] = EnvState.ERROR
            raise e

    def seed(self, seed: Union[List[int], int], dynamic_seed: bool = None) -> None:
        """
        Overview:
            Set the seed for each environment.
        Arguments:
            - seed (:obj:`Union[List[int], int]`): List of seeds for each environment, \
                or one seed for the first environment and other seeds are generated automatically.
        """
        if isinstance(seed, numbers.Integral):
            seed = [seed + i for i in range(self.env_num)]
        elif isinstance(seed, list):
            assert len(seed) == self._env_num, "len(seed) {:d} != env_num {:d}".format(len(seed), self._env_num)
            seed = seed
        self._env_seed = seed
        self._env_dynamic_seed = dynamic_seed

    def enable_save_replay(self, replay_path: Union[List[str], str]) -> None:
        if isinstance(replay_path, str):
            replay_path = [replay_path] * self.env_num
        self._env_replay_path = replay_path

    def close(self) -> None:
        """
        Overview:
            Release the environment resources.
        """
        if self._closed:
            return
        self._env_ref.close()
        for env in self._envs:
            env.close()
        for i in range(self._env_num):
            self._env_states[i] = EnvState.VOID
        self._closed = True

    def env_info(self) -> namedtuple:
        return self._env_ref.info()

    def env_default_config(self) -> EasyDict:
        return self._env_ref.default_config()


def create_env_manager(manager_cfg: dict, env_fn: List[Callable]) -> BaseEnvManager:
    manager_cfg = copy.deepcopy(manager_cfg)
    if 'import_names' in manager_cfg:
        import_module(manager_cfg.pop('import_names'))
    manager_type = manager_cfg.pop('type')
    return ENV_MANAGER_REGISTRY.build(manager_type, env_fn=env_fn, cfg=manager_cfg)


def get_env_manager_cls(cfg: EasyDict) -> type:
    import_module(cfg.get('import_names', []))
    return ENV_MANAGER_REGISTRY.get(cfg.type)

from abc import ABC
from types import MethodType
from typing import Union, Any, List, Callable, Iterable, Dict, Optional
from functools import partial
from collections import namedtuple
import numbers
import torch
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from nervex.utils import ENV_MANAGER_REGISTRY, import_module


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

    def __init__(
            self,
            env_fn: Callable,
            env_cfg: Iterable,
            env_num: int,
            episode_num: Optional[Union[int, float]] = float('inf'),
            manager_cfg: Optional[dict] = {},
    ) -> None:
        """
        Overview:
            Initialize the BaseEnvManager.
        Arguments:
            - env_fn (:obj:`function`): the function to create environment
            - env_cfg (:obj:`list`): the list of environemnt configs
            - env_num (:obj:`int`): number of environments to create, equal to len(env_cfg)
            - episode_num (:obj:`Optional[Union[int, float]]`): maximum episodes to collect in one environment
            - manager_cfg (:obj:`Optional[dict]`): config for env manager
        """
        self._env_fn = env_fn
        self._env_cfg = env_cfg
        self._env_num = env_num
        if episode_num == "inf":
            episode_num = float("inf")
        self._episode_num = episode_num
        self._transform = partial(to_ndarray)
        self._inv_transform = partial(to_tensor, dtype=torch.float32)
        self._closed = True
        self._env_replay_path = None
        # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
        self._env_ref = self._env_fn(self._env_cfg[0])

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
        return all([self._env_episode_count[env_id] >= self._episode_num for env_id in range(self.env_num)])

    @property
    def method_name_list(self) -> list:
        return ['reset', 'step', 'seed', 'close', 'enable_save_replay']

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
        self.reset(reset_param)

    def _create_state(self) -> None:
        self._closed = False
        self._env_episode_count = {i: 0 for i in range(self.env_num)}
        self._ready_obs = {i: None for i in range(self.env_num)}
        self._envs = [self._env_fn(c) for c in self._env_cfg]
        assert len(self._envs) == self._env_num
        if self._env_replay_path is not None:
            for e, s in zip(self._envs, self._env_replay_path):
                e.enable_save_replay(s)

    def reset(self, reset_param: List[dict] = None) -> None:
        """
        Overview:
            Reset the environments and hyper-params.
        Arguments:
            - reset_param (:obj:`List`): list of reset parameters for each environment.
        """
        if reset_param is None:
            reset_param = [{} for _ in range(self.env_num)]
        self._reset_param = reset_param
        # set seed
        if hasattr(self, '_env_seed'):
            for env, s in zip(self._envs, self._env_seed):
                env.seed(s)
        for i in range(self.env_num):
            self._reset(i)

    def _reset(self, env_id: int) -> None:
        obs = self._safe_run(lambda: self._envs[env_id].reset(**self._reset_param[env_id]))
        self._ready_obs[env_id] = obs

    def _safe_run(self, fn: Callable) -> Any:
        try:
            return fn()
        except Exception as e:
            self.close()
            raise e

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
            timesteps[env_id] = self._safe_run(lambda: self._envs[env_id].step(act))
            if timesteps[env_id].done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num:
                    self._reset(env_id)
            else:
                self._ready_obs[env_id] = timesteps[env_id].obs
        return self._inv_transform(timesteps)

    def seed(self, seed: Union[List[int], int]) -> None:
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
        self._closed = True

    def env_info(self) -> namedtuple:
        return self._env_ref.info()


def create_env_manager(type_: str, **kwargs) -> BaseEnvManager:
    if 'import_names' in kwargs:
        import_module(kwargs.pop('import_names'))
    return ENV_MANAGER_REGISTRY.build(type_, **kwargs)

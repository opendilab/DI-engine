from abc import ABC
from types import MethodType
from typing import Union, Any, List, Callable, Iterable, Dict, Optional
from functools import partial
from collections import namedtuple
import numbers
import torch
from nervex.torch_utils import to_tensor, to_ndarray, to_list


class BaseEnvManager(ABC):
    """
    Overview:
        Create a BaseEnvManager to manage multiple environments.

    Interfaces:
        seed, launch, next_obs, step, reset, env_info
    """

    def __init__(
            self,
            env_fn: Callable,
            env_cfg: Iterable,
            env_num: int,
            episode_num: Optional[int] = 'inf',
            manager_cfg: Optional[dict] = {},
    ) -> None:
        """
        Overview:
            Initialize the BaseEnvManager.
        Arguments:
            - env_fn (:obj:`function`): the function to create environment
            - env_cfg (:obj:`list`): the list of environemnt configs
            - env_num (:obj:`int`): number of environments to create, equal to len(env_cfg)
            - episode_num (:obj:`int`): maximum episodes to collect in one environment
            - manager_cfg (:obj:`dict`): config for env manager
        """
        self._env_num = env_num
        self._env_fn = env_fn
        self._env_cfg = env_cfg
        if episode_num == 'inf':
            episode_num = float('inf')
        self._epsiode_num = episode_num
        self._transform = partial(to_ndarray)
        self._inv_transform = partial(to_tensor, dtype=torch.float32)
        self._closed = True
        # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
        self._env_ref = self._env_fn(self._env_cfg[0])

    def _create_state(self) -> None:
        self._closed = False
        self._env_episode_count = {i: 0 for i in range(self.env_num)}
        self._env_done = {i: False for i in range(self.env_num)}
        self._next_obs = {i: None for i in range(self.env_num)}
        self._envs = [self._env_fn(c) for c in self._env_cfg]
        assert len(self._envs) == self._env_num

    def _check_closed(self):
        assert not self._closed, "env manager is closed, please use the alive env manager"

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def next_obs(self) -> Dict[int, Any]:
        """
        Overview:
            Get the next observations and corresponding env id.
        Return:
            A dictionary with observations and their environment IDs.
        Note:
            The observations are returned in torch.Tensor.
        Example:
            >>>     obs_dict = env_manager.next_obs
            >>>     action_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
        """
        return self._inv_transform({i: self._next_obs[i] for i, d in self._env_done.items() if not d})

    @property
    def done(self) -> bool:
        return all([v == self._epsiode_num for v in self._env_episode_count.values()])

    @property
    def method_name_list(self) -> list:
        return ['reset', 'step', 'seed', 'close']

    def __getattr__(self, key: str) -> Any:
        """
        Note: if a python object doesn't have the attribute named key, it will call this method
        """
        # we suppose that all the envs has the same attributes, if you need different envs, please
        # create different env managers.
        if not hasattr(self._env_ref, key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._env_ref), key))
        if isinstance(getattr(self._env_ref, key), MethodType) and key not in self.method_name_list:
            raise RuntimeError("env getattr doesn't supports method({}), please override method_name_list".format(key))
        self._check_closed()
        return [getattr(env, key) if hasattr(env, key) else None for env in self._envs]

    def launch(self, reset_param: Union[None, List[dict]] = None) -> None:
        """
        Overview:
            Set up the environments and hyper-params.
        Arguments:
            - reset_param (:obj:`List`): list of reset parameters for each environment.
        """
        assert self._closed, "please first close the env manager"
        self._create_state()
        self.reset(reset_param)

    def reset(self, reset_param: Union[None, List[dict]] = None) -> None:
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
        self._next_obs[env_id] = obs

    def _safe_run(self, fn: Callable):
        try:
            return fn()
        except Exception as e:
            self.close()
            raise e

    def step(self, action: Dict[int, Any]) -> Dict[int, namedtuple]:
        """
        Overview:
            Wrapper of step function in the environment.
        Arguments:
            - action (:obj:`Dict`): a dictionary, {env_id: action}, which includes actions and their env ids.
        Return:
            - timesteps (:obj:`Dict`): a dictionary, {env_id: timestep}, which includes each environment's timestep.
        Note:
            - The env_id that appears in action will also be returned in timesteps.
            - It will wait until all environments are done to reset. If episodes in different environments \
                vary significantly, it is suggested to use vec_env_manager.
        Example:
            >>>     action_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
            >>>     timesteps = env_manager.step(action_dict):
            >>>     for env_id, timestep in timesteps.items():
            >>>         pass
        """
        self._check_closed()
        timesteps = {}
        for env_id, act in action.items():
            act = self._transform(act)
            timesteps[env_id] = self._safe_run(lambda: self._envs[env_id].step(act))
            if timesteps[env_id].done:
                self._env_done[env_id] = True
                self._env_episode_count[env_id] += 1
            self._next_obs[env_id] = timesteps[env_id].obs
        if not self.done and all([d for d in self._env_done.values()]):
            for i in range(self.env_num):
                self._reset(i)
                self._env_done[i] = False
        return self._inv_transform(timesteps)

    def seed(self, seed: Union[List[int], int]) -> None:
        """
        Overview:
            Set the seed for each environment.
        Arguments:
            - seed (:obj:`List or int`): list of seeds for each environment, \
                or one seed for the first environment and other seeds are generated automatically.
        """
        if isinstance(seed, numbers.Integral):
            seed = [seed + i for i in range(self.env_num)]
        elif isinstance(seed, list):
            assert len(seed) == self._env_num, "len(seed) {:d} != env_num {:d}".format(len(seed), self._env_num)
            seed = seed
        self._env_seed = seed

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

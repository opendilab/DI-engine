from types import MethodType
from typing import Union, Any, List, Callable, Dict, Optional, Tuple
from functools import partial, wraps
from easydict import EasyDict
import copy
import platform
import numbers
from ditk import logging
import enum
import time
import treetensor.numpy as tnp
from ding.utils import ENV_MANAGER_REGISTRY, import_module, one_time_warning, make_key_as_identifier, WatchDog, \
    remove_illegal_item
from ding.envs.env import BaseEnvTimestep


class EnvState(enum.IntEnum):
    VOID = 0
    INIT = 1
    RUN = 2
    RESET = 3
    DONE = 4
    ERROR = 5
    NEED_RESET = 6


def timeout_wrapper(func: Callable = None, timeout: Optional[int] = None) -> Callable:
    """
    Overview:
        Watch the function that must be finihsed within a period of time. If timeout, raise the captured error.
    """
    if func is None:
        return partial(timeout_wrapper, timeout=timeout)
    if timeout is None:
        return func

    windows_flag = platform.system().lower() == 'windows'
    if windows_flag:
        one_time_warning("Timeout wrapper is not implemented in windows platform, so ignore it default")
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        watchdog = WatchDog(timeout)
        try:
            watchdog.start()
        except ValueError as e:
            # watchdog invalid case
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
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
        reset, step, seed, close, enable_save_replay, launch, default_config, env_state_done
    Properties:
        env_num, ready_obs, done, method_name_list
        observation_space, action_space, reward_space
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        episode_num=float("inf"),
        max_retry=1,
        retry_type='reset',
        auto_reset=True,
        step_timeout=None,
        reset_timeout=None,
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
            - env_fn (:obj:`List[Callable]`): The function to create environment
            - cfg (:obj:`EasyDict`): Config
        """
        self._cfg = cfg
        self._env_fn = env_fn
        self._env_num = len(self._env_fn)
        self._closed = True
        self._env_replay_path = None
        # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
        self._env_ref = self._env_fn[0]()
        try:
            self._observation_space = self._env_ref.observation_space
            self._action_space = self._env_ref.action_space
            self._reward_space = self._env_ref.reward_space
        except:
            # For some environment,
            # we have to reset before getting observation description.
            # However, for dmc-mujoco, we should not reset the env at the main thread,
            # when using in a subprocess mode, which would cause opengl rendering bugs,
            # leading to no response subprocesses.
            self._env_ref.reset()
            self._observation_space = self._env_ref.observation_space
            self._action_space = self._env_ref.action_space
            self._reward_space = self._env_ref.reward_space
            self._env_ref.close()
        self._env_states = {i: EnvState.VOID for i in range(self._env_num)}
        self._env_seed = {i: None for i in range(self._env_num)}

        self._episode_num = self._cfg.episode_num
        self._max_retry = max(self._cfg.max_retry, 1)
        self._auto_reset = self._cfg.auto_reset
        self._retry_type = self._cfg.retry_type
        assert self._retry_type in ['reset', 'renew'], self._retry_type
        self._step_timeout = self._cfg.step_timeout
        self._reset_timeout = self._cfg.reset_timeout
        self._retry_waiting_time = self._cfg.retry_waiting_time

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def env_ref(self) -> int:
        return self._env_ref

    @property
    def observation_space(self) -> 'gym.spaces.Space':  # noqa
        return self._observation_space

    @property
    def action_space(self) -> 'gym.spaces.Space':  # noqa
        return self._action_space

    @property
    def reward_space(self) -> 'gym.spaces.Space':  # noqa
        return self._reward_space

    @property
    def ready_obs(self) -> Dict[int, Any]:
        """
        Overview:
            Get the ready (next) observation, which is uniform for both aysnc/sync scenarios.
        Return:
            - ready_obs (:obj:`Dict[int, Any]:`): Dict with env_id keys and observation values.
        Example:
            >>> obs = env_manager.ready_obs
            >>> stacked_obs = np.concatenate(list(obs.values()))
            >>> action = model(obs)  # model input np obs and output np action
            >>> action = {env_id: a for env_id, a in zip(obs.keys(), action)}
            >>> timesteps = env_manager.step(action)
        """
        active_env = [i for i, s in self._env_states.items() if s == EnvState.RUN]
        return {i: self._ready_obs[i] for i in active_env}

    @property
    def ready_obs_id(self) -> List[int]:
        # In BaseEnvManager, if env_episode_count equals episode_num, this env is done.
        return [i for i, s in self._env_states.items() if s == EnvState.RUN]

    @property
    def ready_imgs(self, render_mode: Optional[str] = 'rgb_array') -> Dict[int, Any]:
        """
        Overview:
            Get the next ready renderd frame and corresponding env id.
        Return:
            - ready_imgs (:obj:`Dict[int, np.ndarray]:`): Dict with env_id keys and rendered frames.
        """
        from ding.utils import render
        assert render_mode in ['rgb_array', 'depth_array']
        return {i: render(self._envs[i], render_mode) for i in self.ready_obs.keys()}

    @property
    def done(self) -> bool:
        return all([s == EnvState.DONE for s in self._env_states.values()])

    @property
    def method_name_list(self) -> list:
        return ['reset', 'step', 'seed', 'close', 'enable_save_replay', 'render']

    def env_state_done(self, env_id: int) -> bool:
        return self._env_states[env_id] == EnvState.DONE

    def __getattr__(self, key: str) -> Any:
        """
        Note:
            If a python object doesn't have the attribute whose name is `key`, it will call this method.
            We suppose that all envs have the same attributes.
            If you need different envs, please implement other env managers.
        """
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

    def launch(self, reset_param: Optional[Dict] = None) -> None:
        """
        Overview:
            Set up the environments and their parameters.
        Arguments:
            - reset_param (:obj:`Optional[Dict]`): Dict of reset parameters for each environment, key is the env_id, \
                value is the cooresponding reset parameters.
        """
        assert self._closed, "Please first close the env manager"
        if reset_param is not None:
            assert len(reset_param) == len(self._env_fn)
        self._create_state()
        self.reset(reset_param)

    def _create_state(self) -> None:
        self._env_episode_count = {i: 0 for i in range(self.env_num)}
        self._ready_obs = {i: None for i in range(self.env_num)}
        self._envs = [e() for e in self._env_fn]
        assert len(self._envs) == self._env_num
        self._reset_param = {i: {} for i in range(self.env_num)}
        self._env_states = {i: EnvState.INIT for i in range(self.env_num)}
        if self._env_replay_path is not None:
            for e, s in zip(self._envs, self._env_replay_path):
                e.enable_save_replay(s)
        self._closed = False

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        """
        Overview:
            Reset the environments their parameters.
        Arguments:
            - reset_param (:obj:`List`): Dict of reset parameters for each environment, key is the env_id, \
                value is the cooresponding reset parameters.
        """
        self._check_closed()
        # set seed if necessary
        env_ids = list(range(self._env_num)) if reset_param is None else list(reset_param.keys())
        for i, env_id in enumerate(env_ids):  # loop-type is necessary
            if self._env_seed[env_id] is not None:
                if self._env_dynamic_seed is not None:
                    self._envs[env_id].seed(self._env_seed[env_id], self._env_dynamic_seed)
                else:
                    self._envs[env_id].seed(self._env_seed[env_id])
                self._env_seed[env_id] = None  # seed only use once
        # reset env
        if reset_param is None:
            env_range = range(self.env_num)
        else:
            for env_id in reset_param:
                self._reset_param[env_id] = reset_param[env_id]
            env_range = reset_param.keys()
        for env_id in env_range:
            if self._env_replay_path is not None and self._env_states[env_id] == EnvState.RUN:
                logging.warning("please don't reset a unfinished env when you enable save replay, we just skip it")
                continue
            self._reset(env_id)

    def _reset(self, env_id: int) -> None:

        @timeout_wrapper(timeout=self._reset_timeout)
        def reset_fn():
            # if self._reset_param[env_id] is None, just reset specific env, not pass reset param
            if self._reset_param[env_id] is not None:
                assert isinstance(self._reset_param[env_id], dict), type(self._reset_param[env_id])
                return self._envs[env_id].reset(**self._reset_param[env_id])
            else:
                return self._envs[env_id].reset()

        exceptions = []
        for _ in range(self._max_retry):
            try:
                self._env_states[env_id] = EnvState.RESET
                obs = reset_fn()
                self._ready_obs[env_id] = obs
                self._env_states[env_id] = EnvState.RUN
                return
            except BaseException as e:
                if self._retry_type == 'renew':
                    err_env = self._envs[env_id]
                    err_env.close()
                    self._envs[env_id] = self._env_fn[env_id]()
                exceptions.append(e)
                time.sleep(self._retry_waiting_time)
                continue

        self._env_states[env_id] = EnvState.ERROR
        self.close()
        logging.error("Env {} reset has exceeded max retries({})".format(env_id, self._max_retry))
        runtime_error = RuntimeError(
            "Env {} reset has exceeded max retries({}), and the latest exception is: {}".format(
                env_id, self._max_retry, repr(exceptions[-1])
            )
        )
        runtime_error.__traceback__ = exceptions[-1].__traceback__
        raise runtime_error

    def step(self, actions: Dict[int, Any]) -> List[Dict[int, BaseEnvTimestep]]:
        """
        Overview:
            Execute env step according to input actions. And reset an env if done.
        Arguments:
            - actions (:obj:`Dict[int, Any]`): actions came from outer caller like policy
        Returns:
            - timesteps (:obj:`List[Dict[int, BaseEnvTimestep]]`): Each timestep is a BaseEnvTimestep object, \
                usually including observation, reward, done, info.
        Example:
            >>> timesteps = env_manager.step(action)
            >>> for i, timestep in enumerate(timesteps):
            >>>     if timestep.done:
            >>>         print('Env {} is done'.format(timestep.env_id))
        """
        self._check_closed()
        timesteps = {}
        for env_id, act in actions.items():
            timesteps[env_id] = self._step(env_id, act)
            if timesteps[env_id].done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num:
                    if self._auto_reset:
                        self._reset(env_id)
                    else:
                        self._env_states[env_id] = EnvState.NEED_RESET
                else:
                    self._env_states[env_id] = EnvState.DONE
            else:
                self._ready_obs[env_id] = timesteps[env_id].obs
        return timesteps

    def _step(self, env_id: int, act: Any) -> BaseEnvTimestep:

        @timeout_wrapper(timeout=self._step_timeout)
        def step_fn():
            return self._envs[env_id].step(act)

        exceptions = []
        for _ in range(self._max_retry):
            try:
                return step_fn()
            except BaseException as e:
                exceptions.append(e)
        self._env_states[env_id] = EnvState.ERROR
        logging.error("Env {} step has exceeded max retries({})".format(env_id, self._max_retry))
        runtime_error = RuntimeError(
            "Env {} step has exceeded max retries({}), and the latest exception is: {}".format(
                env_id, self._max_retry, repr(exceptions[-1])
            )
        )
        runtime_error.__traceback__ = exceptions[-1].__traceback__
        raise runtime_error

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        """
        Overview:
            Set the seed for each environment.
        Arguments:
            - seed (:obj:`Union[Dict[int, int], List[int], int]`): List of seeds for each environment; \
                Or one seed for the first environment and other seeds are generated automatically.
        """
        if isinstance(seed, numbers.Integral):
            seed = [seed + i for i in range(self.env_num)]
            self._env_seed = seed
        elif isinstance(seed, list):
            assert len(seed) == self._env_num, "len(seed) {:d} != env_num {:d}".format(len(seed), self._env_num)
            self._env_seed = seed
        elif isinstance(seed, dict):
            if not hasattr(self, '_env_seed'):
                raise RuntimeError("please indicate all the seed of each env in the beginning")
            for env_id, s in seed.items():
                self._env_seed[env_id] = s
        else:
            raise TypeError("invalid seed arguments type: {}".format(type(seed)))
        self._env_dynamic_seed = dynamic_seed
        try:
            self._action_space.seed(seed[0])
        except Exception:  # TODO(nyz) deal with nested action_space like SMAC
            pass

    def enable_save_replay(self, replay_path: Union[List[str], str]) -> None:
        """
        Overview:
            Set each env's replay save path.
        Arguments:
            - replay_path (:obj:`Union[List[str], str]`): List of paths for each environment; \
                Or one path for all environments.
        """
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
        for env in self._envs:
            env.close()
        for i in range(self._env_num):
            self._env_states[i] = EnvState.VOID
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


@ENV_MANAGER_REGISTRY.register('base_v2')
class BaseEnvManagerV2(BaseEnvManager):
    """
    Overview:
        BaseEnvManager for new task pipeline and interfaces coupled with treetensor.
    """

    @property
    def ready_obs(self) -> tnp.array:
        """
        Overview:
            Get the ready (next) observation in ``tnp.array`` type, which is uniform for both async/sync scenarios.
        Return:
            - ready_obs (:obj:`tnp.array`): A stacked treenumpy-type observation data.
        Example:
            >>> obs = env_manager.ready_obs
            >>> action = model(obs)  # model input np obs and output np action
            >>> timesteps = env_manager.step(action)
        """
        active_env = [i for i, s in self._env_states.items() if s == EnvState.RUN]
        obs = [self._ready_obs[i] for i in active_env]
        return tnp.stack(obs)

    def step(self, actions: List[tnp.ndarray]) -> List[tnp.ndarray]:
        """
        Overview:
            Execute env step according to input actions. And reset an env if done.
        Arguments:
            - actions (:obj:`List[tnp.ndarray]`): actions came from outer caller like policy
        Returns:
            - timesteps (:obj:`List[tnp.ndarray]`): Each timestep is a tnp.array with observation, reward, done, \
                info, env_id.
        """
        actions = {env_id: a for env_id, a in zip(self.ready_obs_id, actions)}
        timesteps = super().step(actions)
        new_data = []
        for env_id, timestep in timesteps.items():
            obs, reward, done, info = timestep
            # make the type and content of key as similar as identifier,
            # in order to call them as attribute (e.g. timestep.xxx), such as ``TimeLimit.truncated`` in cartpole info
            info = make_key_as_identifier(info)
            info = remove_illegal_item(info)
            new_data.append(tnp.array({'obs': obs, 'reward': reward, 'done': done, 'info': info, 'env_id': env_id}))
        return new_data


def create_env_manager(manager_cfg: dict, env_fn: List[Callable]) -> BaseEnvManager:
    r"""
    Overview:
        Create an env manager according to manager cfg and env function.
    Arguments:
        - manager_cfg (:obj:`EasyDict`): Env manager config.
        - env_fn (:obj:` List[Callable]`): A list of envs' functions.
    ArgumentsKeys:
        - `manager_cfg`'s necessary: `type`
    """
    manager_cfg = copy.deepcopy(manager_cfg)
    if 'import_names' in manager_cfg:
        import_module(manager_cfg.pop('import_names'))
    manager_type = manager_cfg.pop('type')
    return ENV_MANAGER_REGISTRY.build(manager_type, env_fn=env_fn, cfg=manager_cfg)


def get_env_manager_cls(cfg: EasyDict) -> type:
    r"""
    Overview:
        Get an env manager class according to cfg.
    Arguments:
        - cfg (:obj:`EasyDict`): Env manager config.
    ArgumentsKeys:
        - necessary: `type`
    """
    import_module(cfg.get('import_names', []))
    return ENV_MANAGER_REGISTRY.get(cfg.type)

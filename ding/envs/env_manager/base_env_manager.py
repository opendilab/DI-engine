from types import MethodType
from typing import Union, Any, List, Callable, Dict, Optional, Tuple
from functools import partial, wraps
from easydict import EasyDict
from ditk import logging
import copy
import platform
import numbers
import enum
import time
import treetensor.numpy as tnp
from ding.utils import ENV_MANAGER_REGISTRY, import_module, one_time_warning, make_key_as_identifier, WatchDog, \
    remove_illegal_item
from ding.envs import BaseEnv, BaseEnvTimestep

global space_log_flag
space_log_flag = True


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
        The basic class of env manager to manage multiple vectorized environments. BaseEnvManager define all the
        necessary interfaces and derived class must extend this basic class.

        The class is implemented by the pseudo-parallelism (i.e. serial) mechanism, therefore, this class is only
        used in some tiny environments and for debug purpose.
    Interfaces:
        reset, step, seed, close, enable_save_replay, launch, default_config, reward_shaping, enable_save_figure
    Properties:
        env_num, env_ref, ready_obs, ready_obs_id, ready_imgs, done, closed, method_name_list, observation_space, \
        action_space, reward_space
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Return the deepcopyed default config of env manager.
        Returns:
            - cfg (:obj:`EasyDict`): The default config of env manager.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        # (int) The total episode number to be executed, defaults to inf, which means no episode limits.
        episode_num=float("inf"),
        # (int) The maximum retry times when the env is in error state, defaults to 1, i.e. no retry.
        max_retry=1,
        # (str) The retry type when the env is in error state, including ['reset', 'renew'], defaults to 'reset'.
        # The former is to reset the env to the last reset state, while the latter is to create a new env.
        retry_type='reset',
        # (bool) Whether to automatically reset sub-environments when they are done, defaults to True.
        auto_reset=True,
        # (float) WatchDog timeout (second) for ``step`` method, defaults to None, which means no timeout.
        step_timeout=None,
        # (float) WatchDog timeout (second) for ``reset`` method, defaults to None, which means no timeout.
        reset_timeout=None,
        # (float) The interval waiting time for automatically retry mechanism, defaults to 0.1.
        retry_waiting_time=0.1,
    )

    def __init__(
            self,
            env_fn: List[Callable],
            cfg: EasyDict = EasyDict({}),
    ) -> None:
        """
        Overview:
            Initialize the base env manager with callable the env function and the EasyDict-type config. Here we use
            ``env_fn`` to ensure the lazy initialization of sub-environments, which is benetificial to resource
            allocation and parallelism. ``cfg`` is the merged result between the default config of this class
            and user's config.
            This construction function is in lazy-initialization mode, the actual initialization is in ``launch``.
        Arguments:
            - env_fn (:obj:`List[Callable]`): A list of functions to create ``env_num`` sub-environments.
            - cfg (:obj:`EasyDict`): Final merged config.

        .. note::
            For more details about how to merge config, please refer to the system document of DI-engine \
            (`en link <../03_system/config.html>`_).
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
        """
        Overview:
            ``env_num`` is the number of sub-environments in env manager.
        Returns:
            - env_num (:obj:`int`): The number of sub-environments.
        """
        return self._env_num

    @property
    def env_ref(self) -> 'BaseEnv':
        """
        Overview:
            ``env_ref`` is used to acquire some common attributes of env, like obs_shape and act_shape.
        Returns:
            - env_ref (:obj:`BaseEnv`): The reference of sub-environment.
        """
        return self._env_ref

    @property
    def observation_space(self) -> 'gym.spaces.Space':  # noqa
        """
        Overview:
            ``observation_space`` is the observation space of sub-environment, following the format of gym.spaces.
        Returns:
            - observation_space (:obj:`gym.spaces.Space`): The observation space of sub-environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> 'gym.spaces.Space':  # noqa
        """
        Overview:
            ``action_space`` is the action space of sub-environment, following the format of gym.spaces.
        Returns:
            - action_space (:obj:`gym.spaces.Space`): The action space of sub-environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> 'gym.spaces.Space':  # noqa
        """
        Overview:
            ``reward_space`` is the reward space of sub-environment, following the format of gym.spaces.
        Returns:
            - reward_space (:obj:`gym.spaces.Space`): The reward space of sub-environment.
        """
        return self._reward_space

    @property
    def ready_obs(self) -> Dict[int, Any]:
        """
        Overview:
            Get the ready (next) observation, which is a special design to unify both aysnc/sync env manager.
            For each interaction between policy and env, the policy will input the ready_obs and output the action.
            Then the env_manager will ``step`` with the action and prepare the next ready_obs.
        Returns:
            - ready_obs (:obj:`Dict[int, Any]`): A dict with env_id keys and observation values.
        Example:
            >>> obs = env_manager.ready_obs
            >>> stacked_obs = np.concatenate(list(obs.values()))
            >>> action = policy(obs)  # here policy inputs np obs and outputs np action
            >>> action = {env_id: a for env_id, a in zip(obs.keys(), action)}
            >>> timesteps = env_manager.step(action)
        """
        active_env = [i for i, s in self._env_states.items() if s == EnvState.RUN]
        return {i: self._ready_obs[i] for i in active_env}

    @property
    def ready_obs_id(self) -> List[int]:
        """
        Overview:
            Get the ready (next) observation id, which is a special design to unify both aysnc/sync env manager.
        Returns:
            - ready_obs_id (:obj:`List[int]`): A list of env_ids for ready observations.
        """
        # In BaseEnvManager, if env_episode_count equals episode_num, this env is done.
        return [i for i, s in self._env_states.items() if s == EnvState.RUN]

    @property
    def ready_imgs(self, render_mode: Optional[str] = 'rgb_array') -> Dict[int, Any]:
        """
        Overview:
            Sometimes, we need to render the envs, this function is used to get the next ready renderd frame and \
            corresponding env id.
        Arguments:
            - render_mode (:obj:`Optional[str]`): The render mode, can be 'rgb_array' or 'depth_array', which follows \
                the definition in the ``render`` function of ``ding.utils`` .
        Returns:
            - ready_imgs (:obj:`Dict[int, np.ndarray]`): A dict with env_id keys and rendered frames.
        """
        from ding.utils import render
        assert render_mode in ['rgb_array', 'depth_array'], render_mode
        return {i: render(self._envs[i], render_mode) for i in self.ready_obs_id}

    @property
    def done(self) -> bool:
        """
        Overview:
            ``done`` is a flag to indicate whether env manager is done, i.e., whether all sub-environments have \
            executed enough episodes.
        Returns:
            - done (:obj:`bool`): Whether env manager is done.
        """
        return all([s == EnvState.DONE for s in self._env_states.values()])

    @property
    def method_name_list(self) -> list:
        """
        Overview:
            The public methods list of sub-environments that can be directly called from the env manager level. Other \
            methods and attributes will be accessed with the ``__getattr__`` method.
            Methods defined in this list can be regarded as the vectorized extension of methods in sub-environments.
            Sub-class of ``BaseEnvManager`` can override this method to add more methods.
        Returns:
            - method_name_list (:obj:`list`): The public methods list of sub-environments.
        """
        return [
            'reset', 'step', 'seed', 'close', 'enable_save_replay', 'render', 'reward_shaping', 'enable_save_figure'
        ]

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
            Launch the env manager, instantiate the sub-environments and set up the environments and their parameters.
        Arguments:
            - reset_param (:obj:`Optional[Dict]`): A dict of reset parameters for each environment, key is the env_id, \
                value is the corresponding reset parameter, defaults to None.
        """
        assert self._closed, "Please first close the env manager"
        try:
            global space_log_flag
            if space_log_flag:
                logging.info("Env Space Information:")
                logging.info("\tObservation Space: {}".format(self._observation_space))
                logging.info("\tAction Space: {}".format(self._action_space))
                logging.info("\tReward Space: {}".format(self._reward_space))
                space_log_flag = False
        except:
            pass
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
            Forcely reset the sub-environments their corresponding parameters. Because in env manager all the \
            sub-environments usually are reset automatically as soon as they are done, this method is only called when \
            the caller must forcely reset all the sub-environments, such as in evaluation.
        Arguments:
            - reset_param (:obj:`List`): Dict of reset parameters for each environment, key is the env_id, \
                value is the corresponding reset parameters.
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
                env_id, self._max_retry, str(exceptions[-1])
            )
        )
        runtime_error.__traceback__ = exceptions[-1].__traceback__
        raise runtime_error

    def step(self, actions: Dict[int, Any]) -> Dict[int, BaseEnvTimestep]:
        """
        Overview:
            Execute env step according to input actions. If some sub-environments are done after this execution, \
            they will be reset automatically when ``self._auto_reset`` is True, otherwise they need to be reset when \
            the caller use the ``reset`` method of env manager.
        Arguments:
            - actions (:obj:`Dict[int, Any]`): A dict of actions, key is the env_id, value is corresponding action. \
                action can be any type, it depends on the env, and the env will handle it. Ususlly, the action is \
                a dict of numpy array, and the value is generated by the outer caller like ``policy``.
        Returns:
            - timesteps (:obj:`Dict[int, BaseEnvTimestep]`): Each timestep is a ``BaseEnvTimestep`` object, \
                usually including observation, reward, done, info. Some special customized environments will have \
                the special timestep definition. The length of timesteps is the same as the length of actions in \
                synchronous env manager.
        Example:
            >>> timesteps = env_manager.step(action)
            >>> for env_id, timestep in enumerate(timesteps):
            >>>     if timestep.done:
            >>>         print('Env {} is done'.format(env_id))
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
                env_id, self._max_retry, str(exceptions[-1])
            )
        )
        runtime_error.__traceback__ = exceptions[-1].__traceback__
        raise runtime_error

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        """
        Overview:
            Set the random seed for each environment.
        Arguments:
            - seed (:obj:`Union[Dict[int, int], List[int], int]`): Dict or List of seeds for each environment; \
                If only one seed is provided, it will be used in the same way for all environments.
            - dynamic_seed (:obj:`bool`): Whether to use dynamic seed.

        .. note::
            For more details about ``dynamic_seed``, please refer to the best practice document of DI-engine \
            (`en link <../04_best_practice/random_seed.html>`_).
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
            Enable all environments to save replay video after each episode terminates.
        Arguments:
            - replay_path (:obj:`Union[List[str], str]`): List of paths for each environment; \
                Or one path for all environments.
        """
        if isinstance(replay_path, str):
            replay_path = [replay_path] * self.env_num
        self._env_replay_path = replay_path

    def enable_save_figure(self, env_id: int, figure_path: str) -> None:
        """
        Overview:
            Enable a specific env to save figure (e.g. environment statistics or episode return curve).
        Arguments:
            - figure_path (:obj:`str`): The file directory path for all environments to save figures.
        """
        assert figure_path is not None
        self._env[env_id].enable_save_figure(figure_path)

    def close(self) -> None:
        """
        Overview:
            Close the env manager and release all the environment resources.
        """
        if self._closed:
            return
        for env in self._envs:
            env.close()
        for i in range(self._env_num):
            self._env_states[i] = EnvState.VOID
        self._closed = True

    def reward_shaping(self, env_id: int, transitions: List[dict]) -> List[dict]:
        """
        Overview:
            Execute reward shaping for a specific environment, which is often called when a episode terminates.
        Arguments:
            - env_id (:obj:`int`): The id of the environment to be shaped.
            - transitions (:obj:`List[dict]`): The transition data list of the environment to be shaped.
        Returns:
            - transitions (:obj:`List[dict]`): The shaped transition data list.
        """
        return self._envs[env_id].reward_shaping(transitions)

    @property
    def closed(self) -> bool:
        """
        Overview:
            ``closed`` is a property that returns whether the env manager is closed.
        Returns:
            - closed (:obj:`bool`): Whether the env manager is closed.
        """
        return self._closed


@ENV_MANAGER_REGISTRY.register('base_v2')
class BaseEnvManagerV2(BaseEnvManager):
    """
    Overview:
        The basic class of env manager to manage multiple vectorized environments. BaseEnvManager define all the
        necessary interfaces and derived class must extend this basic class.

        The class is implemented by the pseudo-parallelism (i.e. serial) mechanism, therefore, this class is only
        used in some tiny environments and for debug purpose.

        ``V2`` means this env manager is designed for new task pipeline and interfaces coupled with treetensor.`

    .. note::
        For more details about new task pipeline, please refer to the system document of DI-engine \
        (`en link <../03_system/index.html>`_).
    Interfaces:
        reset, step, seed, close, enable_save_replay, launch, default_config, reward_shaping, enable_save_figure
    Properties:
        env_num, env_ref, ready_obs, ready_obs_id, ready_imgs, done, closed, method_name_list, observation_space, \
        action_space, reward_space
    """

    @property
    def ready_obs(self) -> tnp.array:
        """
        Overview:
            Get the ready (next) observation, which is a special design to unify both aysnc/sync env manager.
            For each interaction between policy and env, the policy will input the ready_obs and output the action.
            Then the env_manager will ``step`` with the action and prepare the next ready_obs.
            For ``V2`` version, the observation is transformed and packed up into ``tnp.array`` type, which allows
            more convenient operations.
        Return:
            - ready_obs (:obj:`tnp.array`): A stacked treenumpy-type observation data.
        Example:
            >>> obs = env_manager.ready_obs
            >>> action = policy(obs)  # here policy inputs treenp obs and output np action
            >>> timesteps = env_manager.step(action)
        """
        active_env = [i for i, s in self._env_states.items() if s == EnvState.RUN]
        obs = [self._ready_obs[i] for i in active_env]
        if isinstance(obs[0], dict):  # transform each element to treenumpy array
            obs = [tnp.array(o) for o in obs]
        return tnp.stack(obs)

    def step(self, actions: List[tnp.ndarray]) -> List[tnp.ndarray]:
        """
        Overview:
            Execute env step according to input actions. If some sub-environments are done after this execution, \
            they will be reset automatically by default.
        Arguments:
            - actions (:obj:`List[tnp.ndarray]`): A list of treenumpy-type actions, the value is generated by the \
                outer caller like ``policy``.
        Returns:
            - timesteps (:obj:`List[tnp.ndarray]`): A list of timestep, Each timestep is a ``tnp.ndarray`` object, \
                usually including observation, reward, done, info, env_id. Some special environments will have \
                the special timestep definition. The length of timesteps is the same as the length of actions in \
                synchronous env manager. For the compatibility of treenumpy, here we use ``make_key_as_identifier`` \
                and ``remove_illegal_item`` functions to modify the original timestep.
        Example:
            >>> timesteps = env_manager.step(action)
            >>> for timestep in timesteps:
            >>>     if timestep.done:
            >>>         print('Env {} is done'.format(timestep.env_id))
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


def create_env_manager(manager_cfg: EasyDict, env_fn: List[Callable]) -> BaseEnvManager:
    """
    Overview:
        Create an env manager according to ``manager_cfg`` and env functions.
    Arguments:
        - manager_cfg (:obj:`EasyDict`): Final merged env manager config.
        - env_fn (:obj:`List[Callable]`): A list of functions to create ``env_num`` sub-environments.
    ArgumentsKeys:
        - type (:obj:`str`): Env manager type set in ``ENV_MANAGER_REGISTRY.register`` , such as ``base`` .
        - import_names (:obj:`List[str]`): A list of module names (paths) to import before creating env manager, such \
            as ``ding.envs.env_manager.base_env_manager`` .
    Returns:
        - env_manager (:obj:`BaseEnvManager`): The created env manager.

    .. tip::
        This method will not modify the ``manager_cfg`` , it will deepcopy the ``manager_cfg`` and then modify it.
    """
    manager_cfg = copy.deepcopy(manager_cfg)
    if 'import_names' in manager_cfg:
        import_module(manager_cfg.pop('import_names'))
    manager_type = manager_cfg.pop('type')
    return ENV_MANAGER_REGISTRY.build(manager_type, env_fn=env_fn, cfg=manager_cfg)


def get_env_manager_cls(cfg: EasyDict) -> type:
    """
    Overview:
        Get the env manager class according to config, which is used to access related class variables/methods.
    Arguments:
        - manager_cfg (:obj:`EasyDict`): Final merged env manager config.
    ArgumentsKeys:
        - type (:obj:`str`): Env manager type set in ``ENV_MANAGER_REGISTRY.register`` , such as ``base`` .
        - import_names (:obj:`List[str]`): A list of module names (paths) to import before creating env manager, such \
            as ``ding.envs.env_manager.base_env_manager`` .
    Returns:
        - env_manager_cls (:obj:`type`): The corresponding env manager class.
    """
    import_module(cfg.get('import_names', []))
    return ENV_MANAGER_REGISTRY.get(cfg.type)

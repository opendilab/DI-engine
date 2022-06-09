from collections import defaultdict
import math
import queue
from time import sleep, time
import gym
from ding.framework import Supervisor
from typing import TYPE_CHECKING, Any, List, Union, Dict, Optional, Callable
from ding.framework.supervisor import ChildType, RecvPayload, SendPayload, SharedObject
from ding.utils import make_key_as_identifier
from ditk import logging
from ding.envs.env_manager.subprocess_env_manager import ShmBufferContainer
import enum
import treetensor.numpy as tnp
import numbers
if TYPE_CHECKING:
    from gym.spaces import Space


class EnvState(enum.IntEnum):
    """
    VOID -> RUN -> DONE
    """
    VOID = 0
    INIT = 1
    RUN = 2
    RESET = 3
    DONE = 4
    ERROR = 5
    NEED_RESET = 6


class EnvRetryType(str, enum.Enum):
    RESET = "reset"
    RENEW = "renew"


class EnvSupervisor(Supervisor):
    """
    Manage multiple envs with supervisor.

    New features (compared to env manager):
    - Consistent interface in multi-process and multi-threaded mode.
    - Add asynchronous features and recommend using asynchronous methods.
    - Reset is performed after an error is encountered in the step method.

    Breaking changes (compared to env manager):
    - Without some states.
    """

    def __init__(
            self,
            type_: ChildType = ChildType.PROCESS,
            env_fn: List[Callable] = None,
            retry_type: EnvRetryType = EnvRetryType.RESET,
            max_try: Optional[int] = None,
            max_retry: Optional[int] = None,
            auto_reset: bool = True,
            reset_timeout: Optional[int] = None,
            step_timeout: Optional[int] = None,
            retry_waiting_time: Optional[int] = None,
            episode_num: int = float("inf"),
            shared_memory: bool = True,
            copy_on_get: bool = True,
            **kwargs
    ) -> None:
        """
        Overview:
            Supervisor that manage a group of envs.
        Arguments:
            - type_ (:obj:`ChildType`): Type of child process.
            - env_fn (:obj:`List[Callable]`): The function to create environment
            - retry_type (:obj:`EnvRetryType`): Retry reset or renew env.
            - max_try (:obj:`EasyDict`): Max try times for reset or step action.
            - max_retry (:obj:`Optional[int]`): Alias of max_try.
            - auto_reset (:obj:`bool`): Auto reset env if reach done.
            - reset_timeout (:obj:`Optional[int]`): Timeout in seconds for reset.
            - step_timeout (:obj:`Optional[int]`): Timeout in seconds for step.
            - retry_waiting_time (:obj:`Optional[float]`): Wait time on each retry.
            - shared_memory (:obj:`bool`): Use shared memory in multiprocessing.
            - copy_on_get (:obj:`bool`): Use copy on get in multiprocessing.
        """
        if kwargs:
            logging.warning("Unknown parameters on env supervisor: {}".format(kwargs))
        super().__init__(type_=type_)
        if type_ is not ChildType.PROCESS and (shared_memory or copy_on_get):
            logging.warning("shared_memory and copy_on_get only works in process mode.")
        self._shared_memory = type_ is ChildType.PROCESS and shared_memory
        self._copy_on_get = type_ is ChildType.PROCESS and copy_on_get
        self._env_fn = env_fn
        self._create_env_ref()
        self._obs_buffers = None
        if env_fn:
            if self._shared_memory:
                obs_space = self._observation_space
                if isinstance(obs_space, gym.spaces.Dict):
                    # For multi_agent case, such as multiagent_mujoco and petting_zoo mpe.
                    # Now only for the case that each agent in the team have the same obs structure
                    # and corresponding shape.
                    shape = {k: v.shape for k, v in obs_space.spaces.items()}
                    dtype = {k: v.dtype for k, v in obs_space.spaces.items()}
                else:
                    shape = obs_space.shape
                    dtype = obs_space.dtype
                self._obs_buffers = {
                    env_id: ShmBufferContainer(dtype, shape, copy_on_get=self._copy_on_get)
                    for env_id in range(len(self._env_fn))
                }
                for env_init in env_fn:
                    self.register(
                        env_init, shared_object=SharedObject(buf=self._obs_buffers, callback=self._shm_callback)
                    )
            else:
                for env_init in env_fn:
                    self.register(env_init)
        self._retry_type = retry_type
        self._auto_reset = auto_reset
        if max_retry:
            logging.warning("The `max_retry` is going to be deprecated, use `max_try` instead!")
        self._max_try = max_try or max_retry or 1
        self._reset_timeout = reset_timeout
        self._step_timeout = step_timeout
        self._retry_waiting_time = retry_waiting_time
        self._env_replay_path = None
        self._episode_num = episode_num
        self._init_states()

    def _init_states(self):
        self._env_seed = {}
        self._env_dynamic_seed = None
        self._env_replay_path = None
        self._env_states = {}
        self._reset_param = {}
        self._ready_obs = {}
        self._env_episode_count = {i: 0 for i in range(self.env_num)}
        self._retry_times = defaultdict(lambda: 0)
        self._last_called = defaultdict(lambda: {"step": math.inf, "reset": math.inf})

    def _shm_callback(self, payload: RecvPayload, obs_buffers: Any):
        if payload.method == "reset" and payload.data is not None:
            obs_buffers[payload.proc_id].fill(payload.data)
            payload.data = None
        elif payload.method == "step" and payload.data is not None:
            obs_buffers[payload.proc_id].fill(payload.data.obs)
            payload.data._replace(obs=None)

    def _create_env_ref(self):
        # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
        self._env_ref = self._env_fn[0]()
        self._env_ref.reset()
        self._observation_space = self._env_ref.observation_space
        self._action_space = self._env_ref.action_space
        self._reward_space = self._env_ref.reward_space
        self._env_ref.close()

    def step(self, actions: Union[Dict[int, List[Any]], List[Any]], block: bool = True) -> Optional[List[tnp.ndarray]]:
        """
        Overview:
            Execute env step according to input actions. And reset an env if done.
        Arguments:
            - actions (:obj:`List[tnp.ndarray]`): Actions came from outer caller like policy, \
                in structure of {env_id: actions}.
            - block (:obj:`bool`): If block, return timesteps, else return none.
        Returns:
            - timesteps (:obj:`List[tnp.ndarray]`): Each timestep is a tnp.array with observation, reward, done, \
                info, env_id.
        """
        assert not self.closed, "Env supervisor has closed."
        if isinstance(actions, List):
            actions = {i: p for i, p in enumerate(actions)}
        assert actions, "Action is empty!"

        send_payloads = []

        for env_id, act in actions.items():
            payload = SendPayload(proc_id=env_id, method="step", args=[act])
            send_payloads.append(payload)
            self.send(payload)

        if not block:
            # Retrieve the data for these steps from the recv method
            return

        # Wait for all steps returns
        recv_payloads = self.recv_all(
            send_payloads, ignore_err=True, callback=self._recv_callback, timeout=self._step_timeout
        )
        return [payload.data for payload in recv_payloads]

    def recv(self, ignore_err: bool = False) -> RecvPayload:
        """
        Overview:
            Wait for recv payload, this function will block the thread.
        Arguments:
            - ignore_err (:obj:`bool`): If ignore_err is true, payload with error object will be discarded.\
                This option will not catch the exception.
        Returns:
            - recv_payload (:obj:`RecvPayload`): Recv payload.
        """
        self._detect_timeout()
        try:
            payload = super().recv(ignore_err=True, timeout=0.1)
            payload = self._recv_callback(payload=payload)
            if payload.err:
                return self.recv(ignore_err=ignore_err)
            else:
                return payload
        except queue.Empty:
            return self.recv(ignore_err=ignore_err)

    def _detect_timeout(self):
        """
        Overview:
            Try to restart all timeout environments if detected timeout.
        """
        for env_id in self._last_called:
            if self._step_timeout and time() - self._last_called[env_id]["step"] > self._step_timeout:
                payload = RecvPayload(
                    proc_id=env_id, method="step", err=TimeoutError("Step timeout on env {}".format(env_id))
                )
                self._recv_queue.put(payload)
                continue
            if self._reset_timeout and time() - self._last_called[env_id]["reset"] > self._reset_timeout:
                payload = RecvPayload(
                    proc_id=env_id, method="reset", err=TimeoutError("Step timeout on env {}".format(env_id))
                )
                self._recv_queue.put(payload)
                continue

    @property
    def env_num(self) -> int:
        return len(self._children)

    @property
    def observation_space(self) -> 'Space':
        return self._observation_space

    @property
    def action_space(self) -> 'Space':
        return self._action_space

    @property
    def reward_space(self) -> 'Space':
        return self._reward_space

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
        active_env.sort()
        obs = [self._ready_obs.get(i) for i in active_env]
        if len(obs) == 0:
            return tnp.array([])
        return tnp.stack(obs)

    @property
    def ready_obs_id(self) -> List[int]:
        return [i for i, s in self.env_states.items() if s == EnvState.RUN]

    @property
    def done(self) -> bool:
        return all([s == EnvState.DONE for s in self.env_states.values()])

    @property
    def method_name_list(self) -> List[str]:
        return ['reset', 'step', 'seed', 'close', 'enable_save_replay']

    @property
    def env_states(self) -> Dict[int, EnvState]:
        return {env_id: self._env_states.get(env_id) or EnvState.VOID for env_id in range(self.env_num)}

    def env_state_done(self, env_id: int) -> bool:
        return self.env_states[env_id] == EnvState.DONE

    def launch(self, reset_param: Optional[Dict] = None, block: bool = True) -> None:
        """
        Overview:
            Set up the environments and their parameters.
        Arguments:
            - reset_param (:obj:`Optional[Dict]`): Dict of reset parameters for each environment, key is the env_id, \
                value is the cooresponding reset parameters.
            - block (:obj:`block`): Whether will block the process and wait for reset states.
        """
        assert self.closed, "Please first close the env supervisor before launch it"
        if reset_param is not None:
            assert len(reset_param) == self.env_num
        self.start_link()
        self._send_seed(self._env_seed, self._env_dynamic_seed, block=block)
        self.reset(reset_param, block=block)
        self._enable_env_replay()

    def reset(self, reset_param: Optional[Dict[int, List[Any]]] = None, block: bool = True) -> None:
        """
        Overview:
            Reset an environment.
        Arguments:
            - reset_param (:obj:`Optional[Dict[int, List[Any]]]`): Dict of reset parameters for each environment, \
                key is the env_id, value is the cooresponding reset parameters.
            - block (:obj:`block`): Whether will block the process and wait for reset states.
        """
        if not reset_param:
            reset_param = {i: {} for i in range(self.env_num)}
        elif isinstance(reset_param, List):
            reset_param = {i: p for i, p in enumerate(reset_param)}

        send_payloads = []

        for env_id, kw_param in reset_param.items():
            self._reset_param[env_id] = kw_param  # For auto reset
            send_payloads += self._reset(env_id, kw_param=kw_param)

        if not block:
            return

        self.recv_all(send_payloads, ignore_err=True, callback=self._recv_callback, timeout=self._reset_timeout)

    def _recv_callback(
            self, payload: RecvPayload, remain_payloads: Optional[Dict[str, SendPayload]] = None
    ) -> RecvPayload:
        """
        Overview:
            The callback function for each received payload, within this method will modify the state of \
            each environment, replace objects in shared memory, and determine if a retry is needed due to an error.
        Arguments:
            - payload (:obj:`RecvPayload`): The received payload.
            - remain_payloads (:obj:`Optional[Dict[str, SendPayload]]`): The callback may be called many times \
                until remain_payloads be cleared, you can append new payload into remain_payloads to call this \
                callback recursively.
        """
        self._set_shared_obs(payload=payload)
        self.change_state(payload=payload)
        if payload.method == "reset":
            return self._recv_reset_callback(payload=payload, remain_payloads=remain_payloads)
        elif payload.method == "step":
            return self._recv_step_callback(payload=payload, remain_payloads=remain_payloads)
        return payload

    def _set_shared_obs(self, payload: RecvPayload):
        if self._obs_buffers is None:
            return
        if payload.method == "reset" and payload.err is None:
            payload.data = self._obs_buffers[payload.proc_id].get()
        elif payload.method == "step" and payload.err is None:
            payload.data._replace(obs=self._obs_buffers[payload.proc_id].get())

    def _recv_reset_callback(
            self, payload: RecvPayload, remain_payloads: Optional[Dict[str, SendPayload]] = None
    ) -> RecvPayload:
        assert payload.method == "reset", "Recv error callback({}) in reset callback!".format(payload.method)
        if remain_payloads is None:
            remain_payloads = {}
        env_id = payload.proc_id
        if payload.err:
            self._retry_times[env_id] += 1
            if self._retry_times[env_id] > self._max_try - 1:
                self.shutdown(5)
                raise RuntimeError(
                    "Env {} reset has exceeded max_try({}), and the latest exception is: {}".format(
                        env_id, self._max_try, payload.err
                    )
                )
            if self._retry_waiting_time:
                sleep(self._retry_waiting_time)
            if self._retry_type == EnvRetryType.RENEW:
                self._children[env_id].restart()
            send_payloads = self._reset(env_id)
            for p in send_payloads:
                remain_payloads[p.req_id] = p
        else:
            self._retry_times[env_id] = 0
            self._ready_obs[env_id] = payload.data
        return payload

    def _recv_step_callback(
            self, payload: RecvPayload, remain_payloads: Optional[Dict[str, SendPayload]] = None
    ) -> RecvPayload:
        assert payload.method == "step", "Recv error callback({}) in step callback!".format(payload.method)
        if remain_payloads is None:
            remain_payloads = {}
        if payload.err:
            send_payloads = self._reset(payload.proc_id)
            for p in send_payloads:
                remain_payloads[p.req_id] = p
            info = {"abnormal": True, "err": payload.err}
            payload.data = tnp.array(
                {
                    'obs': None,
                    'reward': None,
                    'done': None,
                    'info': info,
                    'env_id': payload.proc_id
                }
            )
        else:
            obs, reward, done, info, *_ = payload.data
            if done:
                self._env_episode_count[payload.proc_id] += 1
                if self._env_episode_count[payload.proc_id] < self._episode_num and self._auto_reset:
                    send_payloads = self._reset(payload.proc_id)
                    for p in send_payloads:
                        remain_payloads[p.req_id] = p
            # make the type and content of key as similar as identifier,
            # in order to call them as attribute (e.g. timestep.xxx), such as ``TimeLimit.truncated`` in cartpole info
            info = make_key_as_identifier(info)
            payload.data = tnp.array(
                {
                    'obs': obs,
                    'reward': reward,
                    'done': done,
                    'info': info,
                    'env_id': payload.proc_id
                }
            )
            self._ready_obs[payload.proc_id] = obs
        return payload

    def _reset(self, env_id: int, kw_param: Optional[Dict[str, Any]] = None) -> List[SendPayload]:
        """
        Overview:
            Reset an environment. This method does not wait for the result to be returned.
        Arguments:
            - env_id (:obj:`int`): Environment id.
            - kw_param (:obj:`Optional[Dict[str, Any]]`): Reset parameters for the environment.
        Returns:
            - send_payloads (:obj:`List[SendPayload]`): The request payloads for seed and reset actions.
        """
        assert not self.closed, "Env supervisor has closed."
        send_payloads = []
        kw_param = kw_param or self._reset_param[env_id]

        if self._env_replay_path is not None and self.env_states[env_id] == EnvState.RUN:
            logging.warning("Please don't reset an unfinished env when you enable save replay, we just skip it")
            return send_payloads

        # Reset env
        payload = SendPayload(proc_id=env_id, method="reset", kwargs=kw_param)
        send_payloads.append(payload)
        self.send(payload)

        return send_payloads

    def _send_seed(self, env_seed: Dict[int, int], env_dynamic_seed: Optional[bool] = None, block: bool = True) -> None:
        send_payloads = []
        for env_id, seed in env_seed.items():
            if seed is None:
                continue
            args = [seed]
            if env_dynamic_seed is not None:
                args.append(env_dynamic_seed)
            payload = SendPayload(proc_id=env_id, method="seed", args=args)
            send_payloads.append(payload)
            self.send(payload)
        if not block or not send_payloads:
            return
        self.recv_all(send_payloads, ignore_err=True, callback=self._recv_callback, timeout=self._reset_timeout)

    def change_state(self, payload: RecvPayload):
        self._last_called[payload.proc_id][payload.method] = math.inf  # Have recevied
        if payload.err:
            self._env_states[payload.proc_id] = EnvState.ERROR
        elif payload.method == "reset":
            self._env_states[payload.proc_id] = EnvState.RUN
        elif payload.method == "step":
            if payload.data[2]:
                self._env_states[payload.proc_id] = EnvState.DONE

    def send(self, payload: SendPayload) -> None:
        self._last_called[payload.proc_id][payload.method] = time()
        return super().send(payload)

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: Optional[bool] = None) -> None:
        """
        Overview:
            Set the seed for each environment. The seed function will not be called until supervisor.launch \
            was called.
        Arguments:
            - seed (:obj:`Union[Dict[int, int], List[int], int]`): List of seeds for each environment; \
                Or one seed for the first environment and other seeds are generated automatically. \
                Note that in threading mode, no matter how many seeds are given, only the last one will take effect. \
                Because the execution in the thread is asynchronous, the results of each experiment \
                are different even if a fixed seed is used.
            - dynamic_seed (:obj:`Optional[bool]`): Dynamic seed is used in the training environment, \
                trying to make the random seed of each episode different, they are all generated in the reset \
                method by a random generator 100 * np.random.randint(1 , 1000) (but the seed of this random \
                number generator is fixed by the environmental seed method, guranteeing the reproducibility \
                of the experiment). You need not pass the dynamic_seed parameter in the seed method, or pass \
                the parameter as True.
        """
        self._env_seed = {}
        if isinstance(seed, numbers.Integral):
            self._env_seed = {i: seed + i for i in range(self.env_num)}
        elif isinstance(seed, list):
            assert len(seed) == self.env_num, "len(seed) {:d} != env_num {:d}".format(len(seed), self.env_num)
            self._env_seed = {i: _seed for i, _seed in enumerate(seed)}
        elif isinstance(seed, dict):
            self._env_seed = {env_id: s for env_id, s in seed.items()}
        else:
            raise TypeError("Invalid seed arguments type: {}".format(type(seed)))
        self._env_dynamic_seed = dynamic_seed

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

    def _enable_env_replay(self):
        if self._env_replay_path is None:
            return
        send_payloads = []
        for env_id, s in enumerate(self._env_replay_path):
            payload = SendPayload(proc_id=env_id, method="enable_save_replay", args=[s])
            send_payloads.append(payload)
            self.send(payload)
        self.recv_all(send_payloads=send_payloads)

    def __getattr__(self, key: str) -> List[Any]:
        if not hasattr(self._env_ref, key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._env_ref), key))
        return super().__getattr__(key)

    def close(self, timeout: Optional[float] = None) -> None:
        """
        In order to be compatible with BaseEnvManager, the new version can use `shutdown` directly.
        """
        self.shutdown(timeout=timeout)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        if self._running:
            send_payloads = []
            for env_id in range(self.env_num):
                payload = SendPayload(proc_id=env_id, method="close")
                send_payloads.append(payload)
                self.send(payload)
            self.recv_all(send_payloads=send_payloads, ignore_err=True, timeout=timeout)
            super().shutdown(timeout=timeout)
            self._init_states()

    @property
    def closed(self) -> bool:
        return not self._running

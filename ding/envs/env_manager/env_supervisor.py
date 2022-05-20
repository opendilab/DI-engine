from time import sleep
from ding.framework import Supervisor
from typing import TYPE_CHECKING, Any, List, Union, Dict, Optional, Callable
from ding.framework.supervisor import ChildType, RecvPayload, SendPayload
from ding.utils import make_key_as_identifier
from ditk import logging
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
            **kwargs
    ) -> None:
        super().__init__(type_=type_)
        if env_fn:
            for env_init in env_fn:
                self.register(env_init)
        self._env_seed = {}
        self._env_dynamic_seed = None
        self._env_replay_path = None
        self._env_states = {}
        self._retry_type = retry_type
        self._reset_param = {}
        self._auto_reset = auto_reset
        self._ready_obs = {}
        if max_retry:
            logging.warning("The `max_retry` is going to be deprecated, use `max_try` instead!")
        self._max_try = max_try or max_retry or 1
        self._reset_timeout = reset_timeout
        self._step_timeout = step_timeout
        self._retry_waiting_time = retry_waiting_time

    def step(self, actions: Optional[Dict[int, List[Any]]], block: bool = True) -> Optional[List[tnp.ndarray]]:
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
            send_payloads, ignore_err=True, callback=self._recv_step_callback(), timeout=self._step_timeout
        )
        return [payload.data for payload in recv_payloads]

    def _recv_step_callback(self) -> Callable:
        reset_callback = self._recv_reset_callback()

        def step_callback(payload: RecvPayload, remain_payloads: Dict[str, SendPayload]):
            self.change_state(payload)
            reset_callback(payload, remain_payloads)
            if payload.method != "step":
                return
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
                obs, reward, done, info = payload.data
                if done and self._auto_reset:
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

        return step_callback

    @property
    def env_num(self) -> int:
        return len(self._children)

    @property
    def observation_space(self) -> 'Space':
        pass

    @property
    def action_space(self) -> 'Space':
        pass

    @property
    def reward_space(self) -> 'Space':
        pass

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
        obs = [self._ready_obs.get(i) for i in active_env]
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
        pass

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
        self.reset(reset_param, block=block)

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

        self.recv_all(send_payloads, ignore_err=True, callback=self._recv_reset_callback(), timeout=self._reset_timeout)

    def _recv_reset_callback(self) -> Callable:
        retry_times = {env_id: 0 for env_id in range(self.env_num)}

        def reset_callback(payload: RecvPayload, remain_payloads: Dict[str, SendPayload]):
            self.change_state(payload)
            if payload.method != "reset":
                return
            env_id = payload.proc_id
            if payload.err:
                retry_times[env_id] += 1
                if retry_times[env_id] > self._max_try - 1:
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
                self._ready_obs[env_id] = payload.data

        return reset_callback

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

        # Set seed if necessary
        seed = self._env_seed.get(env_id)
        if seed is not None:
            args = [seed]
            if self._env_dynamic_seed is not None:
                args.append(self._env_dynamic_seed)
            payload = SendPayload(proc_id=env_id, method="seed", args=args)
            send_payloads.append(payload)
            self.send(payload)

        # Reset env
        payload = SendPayload(proc_id=env_id, method="reset", kwargs=kw_param)
        send_payloads.append(payload)
        self.send(payload)

        return send_payloads

    def change_state(self, payload: RecvPayload):
        if payload.err:
            self._env_states[payload.proc_id] = EnvState.ERROR
        elif payload.method == "reset":
            self._env_states[payload.proc_id] = EnvState.RUN
        elif payload.method == "step":
            if payload.data[2]:
                self._env_states[payload.proc_id] = EnvState.DONE

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        """
        Overview:
            Set the seed for each environment.
        Arguments:
            - seed (:obj:`Union[Dict[int, int], List[int], int]`): List of seeds for each environment; \
                Or one seed for the first environment and other seeds are generated automatically.
            - dynamic_seed (:obj:`bool`): Dynamic seed is used in the training environment, \
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
        pass

    def close(self, timeout: Optional[float] = None) -> None:
        """
        In order to be compatible with BaseEnvManager, the new version can use `shutdown` directly.
        """
        self.shutdown(timeout=timeout)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        if self._running:
            for env_id in range(self.env_num):
                self.send(SendPayload(proc_id=env_id, method="close"))
            super().shutdown(timeout=timeout)
            self._env_states = {}

    @property
    def closed(self) -> bool:
        return not self._running

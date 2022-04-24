from ding.framework import Supervisor
from typing import TYPE_CHECKING, Any, List, Union, Dict, Optional, Callable
from ding.framework.supervisor import ChildType, RecvPayload, SendPayload
from ding.utils import make_key_as_identifier
import enum
import treetensor.numpy as tnp
import numbers
import logging
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
    1. Consistent interface in multi-process and multi-threaded mode.
    2. Add asynchronous features and recommend using asynchronous methods.
    3. Automatic recovery of error-prone environments.

    Breaking changes (compared to env manager):
    1. Without properties from __getattr__.
    2. Without some states.
    3. Change auto_reset to a step feature instead of a global config.
    """

    def __init__(
            self,
            type_: ChildType = ChildType.PROCESS,
            env_fn: List[Callable] = None,
            retry_type: EnvRetryType = EnvRetryType.RESET,
            max_try: int = None,
            max_retry: int = None,
            **kwargs
    ) -> None:
        super().__init__(type_=type_)
        if env_fn:
            for env_init in env_fn:
                self.register(env_init)
        self._closed = True
        self._env_seed = {}
        self._env_dynamic_seed = None
        self._env_replay_path = None
        self._env_states = {}
        self._retry_type = retry_type
        self._reset_param = {}
        if max_retry:
            logging.warning("The `max_retry` is going to be deprecated, use `max_try` instead!")
        self._max_try = max_try or max_retry or 1

    def step(self, actions: Dict[int, Any], block: bool = True, auto_reset: bool = True) -> Optional[List[tnp.ndarray]]:
        assert not self.closed, "Env supervisor has closed."

        req_ids = []

        for env_id, act in actions.items():
            payload = SendPayload(proc_id=env_id, method="step", args=[act])
            req_ids.append(payload.req_id)
            self.send(payload)

        if not block:
            # Retrieve the data for these steps from the recv method
            return

        recv_payloads = self.recv_all(req_ids, ignore_err=True)

        new_data = []
        for payload in recv_payloads:
            self.change_state(payload)
            if payload.err:
                info = {"abnormal": True, "err": payload.err}
                new_data.append(
                    {tnp.array({
                        'obs': None,
                        'reward': None,
                        'done': None,
                        'info': info,
                        'env_id': payload.proc_id
                    })}
                )
            else:
                obs, reward, done, info = payload.data
                # make the type and content of key as similar as identifier,
                # in order to call them as attribute (e.g. timestep.xxx), such as ``TimeLimit.truncated`` in cartpole info
                info = make_key_as_identifier(info)
                new_data.append(
                    tnp.array({
                        'obs': obs,
                        'reward': reward,
                        'done': done,
                        'info': info,
                        'env_id': payload.proc_id
                    })
                )
        return new_data

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
        pass

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

        req_ids = []
        req_seqs = {}

        def reset_fn(env_id, kw_param, max_try):
            for _ in range(max_try):
                yield self._reset(env_id, kw_param)

        for env_id, kw_param in reset_param.items():
            self._reset_param[env_id] = kw_param  # For auto reset
            g = reset_fn(env_id, kw_param, max_try=self._max_try)
            req_seqs[env_id] = g
            req_ids += next(g)

        # Try max_try times in blocking mode
        if block:
            while req_ids:
                payload = self.recv(ignore_err=True)
                if payload.req_id not in req_ids:
                    self._recv_queue.put(payload)
                    continue
                self.change_state(payload)
                req_ids.remove(payload.req_id)
                if payload.method == "reset" and payload.err:
                    try:
                        req_ids += next(req_seqs[payload.proc_id])
                    except StopIteration:
                        raise RuntimeError(
                            "Env {} reset has exceeded max retries({}), and the latest exception is: {}".format(
                                payload.proc_id, self._max_try, payload.err
                            )
                        )

    def _reset(self, env_id: int, kw_param: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Overview:
            Reset an environment.
        Arguments:
            - env_id (:obj:`int`): Environment id.
            - kw_param (:obj:`Optional[Dict[str, Any]]`): Reset parameters for the environment.
        Returns:
            - req_ids (:obj:`List[str]`): The request ids for seed and reset actions.
        """
        assert not self.closed, "Env supervisor has closed."
        req_ids = []
        kw_param = kw_param or {}

        if self._env_replay_path is not None and self.env_states[env_id] == EnvState.RUN:
            logging.warning("Please don't reset an unfinished env when you enable save replay, we just skip it")
            return req_ids

        # Set seed if necessary
        seed = self._env_seed.get(env_id)
        if seed is not None:
            args = [seed]
            if self._env_dynamic_seed is not None:
                args.append(self._env_dynamic_seed)
            payload = SendPayload(proc_id=env_id, method="seed", args=args)
            req_ids.append(payload.req_id)
            self.send(payload)

        # Reset env
        payload = SendPayload(proc_id=env_id, method="reset", kwargs=kw_param)
        req_ids.append(payload.req_id)
        self.send(payload)

        return req_ids

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

    def close(self) -> None:
        """
        In order to be compatible with BaseEnvManager, the new version can use `shutdown` directly.
        """
        self.shutdown()

    def shutdown(self) -> None:
        if not self._closed:
            for env_id in range(self.env_num):
                self.send(SendPayload(proc_id=env_id, method="close"))
            super().shutdown()
            self._closed = True
            self._env_states = {}

    def start_link(self) -> None:
        if self._closed:
            super().start_link()
            self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

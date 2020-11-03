from multiprocessing import Process, Pipe, connection
from collections import namedtuple
from threading import Thread
import enum
import time
import math
import traceback
from types import MethodType
from typing import Any, Union, List, Tuple, Iterable, Dict

import cloudpickle

from .base_env_manager import BaseEnvManager


class EnvState(enum.IntEnum):
    INIT = 1
    RUN = 2
    RESET = 3
    DONE = 4


class CloudpickleWrapper(object):
    """
    Overview:
        CloudpickleWrapper can be able to pickle more python object(e.g: a object with lambda expression)
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        self.data = cloudpickle.loads(data)


class SubprocessEnvManager(BaseEnvManager):

    def __init__(self, *args, **kwargs) -> None:
        super(SubprocessEnvManager, self).__init__(*args, **kwargs)
        self._parent_remote, self._child_remote = zip(*[Pipe() for _ in range(self.env_num)])
        self._processes = [
            Process(
                target=self.worker_fn,
                args=(parent, child, CloudpickleWrapper(lambda: self._env_fn(cfg)), self.method_name_list),
                daemon=True
            ) for parent, child, cfg in zip(self._parent_remote, self._child_remote, self._env_cfg)
        ]
        for p in self._processes:
            p.start()
        for c in self._child_remote:
            c.close()
        self._env_state = {i: EnvState.INIT for i in range(self.env_num)}
        self._waiting_env = {'step': set()}
        self._setup_async_args()

    def _setup_async_args(self) -> None:
        self._async_args = {
            'step': {
                'wait_num': 2,
                'timeout': 1.5
            },
        }

    @property
    def active_env(self) -> List[int]:
        return [i for i, s in self._env_state.items() if s == EnvState.RUN]

    @property
    def ready_env(self) -> List[int]:
        return [i for i in self.active_env if i not in self._waiting_env['step']]

    @property
    def next_obs(self) -> Dict[int, Any]:
        return {i: self._next_obs[i] for i in self.ready_env}

    @property
    def done(self) -> bool:
        return all([s == EnvState.DONE for s in self._env_state.values()])

    def launch(self, reset_param: Union[None, List[dict]] = None) -> None:
        self._check_closed()
        if reset_param is None:
            reset_param = [[] for _ in range(self.env_num)]
        self._reset_param = reset_param
        # set seed
        if hasattr(self, '_env_seed'):
            for i in range(self.env_num):
                self._parent_remote[i].send(CloudpickleWrapper(['seed', [self._env_seed[i]], {}]))
            ret = [p.recv().data for p in self._parent_remote]
            self._check_data(ret)

        # launch env
        for i in range(self.env_num):
            self._parent_remote[i].send(CloudpickleWrapper(['reset', self._reset_param[i], {}]))
            self._env_state[i] = EnvState.RESET
        obs = [p.recv().data for p in self._parent_remote]
        self._check_data(obs)
        for i in range(self.env_num):
            self._env_state[i] = EnvState.RUN
            self._next_obs[i] = obs[i]

    def _reset(self, env_id: int) -> None:
        self._parent_remote[env_id].send(CloudpickleWrapper(['reset', self._reset_param[env_id], {}]))
        obs = self._parent_remote[env_id].recv().data
        self._check_data([obs])
        self._env_state[env_id] = EnvState.RUN
        self._next_obs[env_id] = obs

    def step(self, action: Dict[int, Any]) -> Dict[int, namedtuple]:
        self._check_closed()
        env_id = list(action.keys())
        assert all([self._env_state[idx] == EnvState.RUN for idx in env_id]), '{}/{}'.format(self._env_state, env_id)

        for i, act in action.items():
            self._parent_remote[i].send(CloudpickleWrapper(['step', [act], {}]))

        handle = self._async_args['step']
        wait_num, timeout = min(handle['wait_num'], len(env_id)), handle['timeout']
        rest_env_id = list(set(env_id).union(self._waiting_env['step']))
        rest_conn = [self._parent_remote[i] for i in rest_env_id]
        ready_conn, ready_idx = SubprocessEnvManager.wait(rest_conn, wait_num, timeout)
        ready_env_id = [rest_env_id[idx] for idx in ready_idx]
        ret = {i: c.recv().data for i, c in zip(ready_env_id, ready_conn)}
        self._check_data(ret.values())

        self._waiting_env['step']: set
        for i in rest_env_id:
            if i in ready_env_id:
                if i in self._waiting_env['step']:
                    self._waiting_env['step'].remove(i)
            else:
                self._waiting_env['step'].add(i)
        for idx, timestep in ret.items():
            if timestep.done:
                self._env_episode_count[idx] += 1
                if self._env_episode_count[idx] >= self._epsiode_num:
                    self._env_state[idx] = EnvState.DONE
                else:
                    self._env_state[idx] = EnvState.RESET
                    reset_thread = Thread(target=self._reset, args=(idx, ))
                    reset_thread.daemon = True
                    reset_thread.start()
            else:
                self._next_obs[idx] = timestep.obs

        return ret

    @property
    def method_name_list(self) -> list:
        return ['reset', 'step', 'seed', 'close']

    # this method must be staticmethod, otherwise there will be some resource conflicts(e.g. port or file)
    # env must be created in worker, which is a trick of avoiding env pickle errors.
    @staticmethod
    def worker_fn(p, c, env_fn_wrapper, method_name_list) -> None:
        env_fn = env_fn_wrapper.data
        env = env_fn()
        p.close()
        try:
            while True:
                try:
                    cmd, args, kwargs = c.recv().data
                except EOFError:  # for the case when the pipe has been closed
                    c.close()
                    break
                try:
                    if cmd == 'getattr':
                        ret = getattr(env, args[0])
                    elif cmd in method_name_list:
                        if args is None and kwargs is None:
                            ret = getattr(env, cmd)()
                        else:
                            ret = getattr(env, cmd)(*args, **kwargs)
                    else:
                        raise KeyError("not support env cmd: {}".format(cmd))
                    c.send(CloudpickleWrapper(ret))
                except Exception as e:
                    # when there are some errors in env, worker_fn will send the errors to env manager
                    # directly send error to another process will lose the stack trace, so we create a new Exception
                    c.send(
                        CloudpickleWrapper(
                            e.__class__('\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)))
                        )
                    )
                if cmd == 'close':
                    c.close()
                    break
        except KeyboardInterrupt:
            c.close()

    def _check_data(self, data: Iterable) -> None:
        for d in data:
            if isinstance(d, Exception):
                # when receiving env Exception, env manager will safely close and raise this Exception to caller
                self.close()
                raise d

    # override
    def __getattr__(self, key: str) -> Any:
        self._check_closed()
        # we suppose that all the envs has the same attributes, if you need different envs, please
        # create different env managers.
        if not hasattr(self._env_ref, key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._env_ref), key))
        if isinstance(getattr(self._env_ref, key), MethodType):
            raise TypeError("env getattr doesn't supports method({}), please override method_name_list".format(key))
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['getattr', [key], {}]))
        ret = [p.recv().data for p in self._parent_remote]
        self._check_data(ret)
        return ret

    # override
    def close(self) -> None:
        if self._closed:
            return
        self._env_ref.close()
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['close', None, None]))
        result = [p.recv().data for p in self._parent_remote]
        for p in self._processes:
            p.join()
        for p in self._processes:
            p.terminate()
        self._closed = True

    @staticmethod
    def wait(rest_conn: list, wait_num: int, timeout: Union[None, float] = None) -> Tuple[list, list]:
        """
        Overview:
            wait at least enough(len(ready_conn) >= wait_num) num connection within timeout constraint
            if timeout is None, wait_num == len(ready_conn), means sync mode;
            if timeout is not None, len(ready_conn) >= wait_num when returns;
        """
        assert 1 <= wait_num <= len(rest_conn
                                    ), 'please indicate proper wait_num: <wait_num: {}, rest_conn_num: {}>'.format(
                                        wait_num, len(rest_conn)
                                    )
        rest_conn_set = set(rest_conn)
        ready_conn = set()
        start_time = time.time()
        rest_time = timeout
        while len(rest_conn_set) > 0:
            finish_conn = set(connection.wait(rest_conn_set, timeout=rest_time))
            ready_conn = ready_conn.union(finish_conn)
            rest_conn_set = rest_conn_set.difference(finish_conn)
            if len(ready_conn) >= wait_num and timeout:
                rest_time = timeout - (time.time() - start_time)
                if rest_time <= 0.0:
                    break
        ready_idx = [rest_conn.index(c) for c in ready_conn]
        return list(ready_conn), ready_idx


class SyncSubprocessEnvManager(SubprocessEnvManager):

    def _setup_async_args(self) -> None:
        self._async_args = {
            'step': {
                'wait_num': math.inf,
                'timeout': None,
            },
        }

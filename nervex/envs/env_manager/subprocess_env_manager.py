from multiprocessing import Process, Pipe, connection, get_context, Array
from collections import namedtuple
import enum
import logging
import platform
import time
import math
import copy
import traceback
import threading
import numpy as np
import torch
import ctypes
import pickle
import cloudpickle
from functools import partial
from types import MethodType
from typing import Any, Union, List, Tuple, Iterable, Dict, Callable, Optional

from nervex.torch_utils import to_tensor, to_ndarray, to_list
from nervex.utils import PropagatingThread, LockContextType, LockContext, ENV_MANAGER_REGISTRY
from .base_env_manager import BaseEnvManager

_NTYPE_TO_CTYPE = {
    np.bool: ctypes.c_bool,
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class EnvState(enum.IntEnum):
    INIT = 1
    RUN = 2
    RESET = 3
    DONE = 4


class ShmBuffer():
    """
    Overview:
        Shared Memory Buffer
    """

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        """
        Overview:
            Initialize the buffer.
        Arguments:
            - dtype (:obj:`np.generic`): dtype of the data to limit the size of the buffer.
            - shape (:obj:`Tuple[int]`): shape of the data to limit the size of the buffer.
        """
        self.buffer = Array(_NTYPE_TO_CTYPE[dtype.type], int(np.prod(shape)))
        self.dtype = dtype
        self.shape = shape

    def fill(self, src_arr: np.ndarray) -> None:
        """
        Overview:
            Fill the shared memory buffer with a numpy array. (Replace the original one.)
        Arguments:
            - src_arr (:obj:`np.ndarray`): array to fill the buffer.
        """
        assert isinstance(src_arr, np.ndarray), type(src_arr)
        dst_arr = np.frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)
        with self.buffer.get_lock():
            np.copyto(dst_arr, src_arr)

    def get(self) -> np.ndarray:
        """
        Overview:
            Get the array stored in the buffer.
        Return:
            - copy_data (:obj:`np.ndarray`): A copy of the data stored in the buffer.
        """
        arr = np.frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)
        return arr.copy()


class ShmBufferContainer(object):
    """
    Overview:
        Support dictionary of shared memory buffer.
    """

    def __init__(self, dtype: np.generic, shape: Union[Dict[Any, tuple], tuple]) -> None:
        if isinstance(shape, dict):
            self._data = {k: ShmBufferContainer(dtype, v) for k, v in shape.items()}
        elif isinstance(shape, (tuple, list)):
            self._data = ShmBuffer(dtype, shape)
        else:
            raise RuntimeError("not support shape: {}".format(shape))
        self._shape = shape

    def fill(self, src_arr: Union[Dict[Any, np.ndarray], np.ndarray]) -> None:
        if isinstance(self._shape, dict):
            for k in self._shape.keys():
                self._data[k].fill(src_arr[k])
        elif isinstance(self._shape, (tuple, list)):
            self._data.fill(src_arr)

    def get(self) -> Union[Dict[Any, np.ndarray], np.ndarray]:
        if isinstance(self._shape, dict):
            return {k: self._data[k].get() for k in self._shape.keys()}
        elif isinstance(self._shape, (tuple, list)):
            return self._data.get()


class CloudPickleWrapper(object):
    """
    Overview:
        CloudPickleWrapper can be able to pickle more python object(e.g: an object with lambda expression)
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        self.data = cloudpickle.loads(data)


def retry_wrapper(fn: Callable, max_retry: int = 10) -> Callable:
    """
    Overview:
        Retry the function until exceeding the maximum retry times.
    """

    def wrapper(*args, **kwargs):
        exceptions = []
        for _ in range(max_retry):
            try:
                ret = fn(*args, **kwargs)
                return ret
            except Exception as e:
                exceptions.append(e)
                time.sleep(0.01)
        e_info = ''.join(
            [
                'Retry {} failed from:\n {}\n'.format(i, ''.join(traceback.format_tb(e.__traceback__)) + str(e))
                for i, e in enumerate(exceptions)
            ]
        )
        fn_exception = Exception("Function {} runtime error:\n{}".format(fn, e_info))
        raise RuntimeError("Function {} has exceeded max retries({})".format(fn, max_retry)) from fn_exception

    return wrapper


@ENV_MANAGER_REGISTRY.register('async_subprocess')
class AsyncSubprocessEnvManager(BaseEnvManager):
    """
    Overview:
        Create a AsyncSubprocessEnvManager to manage multiple environments. Each Environment is run by a seperate \
            subprocess.
    Interfaces:
        seed, launch, ready_obs, step, reset, env_info
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
            Initialize the AsyncSubprocessEnvManager.
        Arguments:
            - env_fn (:obj:`function`): the function to create environment
            - env_cfg (:obj:`list`): the list of environemnt configs
            - env_num (:obj:`int`): number of environments to create, equal to len(env_cfg)
            - episode_num (:obj:`int`): maximum episodes to collect in one environment
            - manager_cfg (:obj:`dict`): config for env manager
        """
        super().__init__(env_fn, env_cfg, env_num, episode_num)
        self.shared_memory = manager_cfg.get('shared_memory', True)
        default_context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        self.context_str = manager_cfg.get('context', default_context_str)
        self.timeout = manager_cfg.get('timeout', 0.01)
        self.wait_num = manager_cfg.get('wait_num', 2)
        self.reset_timeout = manager_cfg.get('reset_timeout', 60)

    def _create_state(self) -> None:
        r"""
        Overview:
            Fork/spawn sub-processes and create pipes to transfer the data.
        """
        self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
        self._ready_obs = {env_id: None for env_id in range(self.env_num)}
        if self.shared_memory:
            obs_space = self._env_ref.info().obs_space
            shape = obs_space.shape
            dtype = np.dtype(obs_space.value['dtype']) if obs_space.value is not None else np.dtype(np.float32)
            self._obs_buffers = {env_id: ShmBufferContainer(dtype, shape) for env_id in range(self.env_num)}
        else:
            self._obs_buffers = {env_id: None for env_id in range(self.env_num)}
        self._pipe_parents, self._pipe_children = {}, {}
        self._subprocesses = {}
        self._env_states = {}
        for env_id in range(self.env_num):
            self._create_env_subprocess(env_id)
        self._waiting_env = {'step': set()}
        self._setup_async_args()
        self._closed = False

    def _create_env_subprocess(self, env_id):
        # start a new one
        env_fn = partial(self._env_fn, cfg=self._env_cfg[env_id])
        self._pipe_parents[env_id], self._pipe_children[env_id] = Pipe()
        ctx = get_context(self.context_str)
        self._subprocesses[env_id] = ctx.Process(
            target=self.worker_fn,
            args=(
                self._pipe_parents[env_id], self._pipe_children[env_id], CloudPickleWrapper(env_fn),
                self._obs_buffers[env_id], self.method_name_list
            ),
            daemon=True,
            name='subprocess_env_manager{}_{}'.format(env_id, time.time())
        )
        self._subprocesses[env_id].start()
        self._pipe_children[env_id].close()
        self._env_states[env_id] = EnvState.INIT

        if self._env_replay_path is not None:
            self._pipe_parents[env_id].send(
                ['enable_save_replay', [self._env_replay_path[env_id]], {}]
            )
            self._pipe_parents[env_id].recv()

    def _setup_async_args(self) -> None:
        r"""
        Overview:
            Set up the async arguments utilized in method ``step``.
            wait_num: for each time the minimum number of env return to gather
            timeout: for each time the minimum number of env return to gather
        """
        self._async_args = {
            'step': {
                'wait_num': self.wait_num,
                'timeout': self.timeout
            },
        }

    @property
    def active_env(self) -> List[int]:
        return [i for i, s in self._env_states.items() if s == EnvState.RUN]

    @property
    def ready_env(self) -> List[int]:
        return [i for i in self.active_env if i not in self._waiting_env['step']]

    @property
    def ready_obs(self) -> Dict[int, Any]:
        """
        Overview:
            Get the next observations.
        Return:
            A dictionary with observations and their environment IDs.
        Note:
            The observations are returned in torch.Tensor.
        Example:
            >>>     obs_dict = env_manager.ready_obs
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
        """
        no_done_env_idx = [i for i, s in self._env_states.items() if s != EnvState.DONE]
        sleep_count = 0
        while all([self._env_states[i] == EnvState.RESET for i in no_done_env_idx]):
            if sleep_count % 1000 == 0:
                logging.warning(
                    'VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count)
                )
            time.sleep(0.001)
            sleep_count += 1
        return {i: self._ready_obs[i] for i in self.ready_env}

    @property
    def done(self) -> bool:
        return all([s == EnvState.DONE for s in self._env_states.values()])

    def launch(self, reset_param: Optional[List[dict]] = None) -> None:
        """
        Overview:
            Set up the environments and hyper-params.
        Arguments:
            - reset_param (:obj:`List`): list of reset parameters for each environment.
        """
        assert self._closed, "please first close the env manager"
        self._create_state()
        self.reset(reset_param)

    def reset(self, reset_param: Optional[List[dict]] = None) -> None:
        """
        Overview:
            Reset the environments and hyper-params.
        Arguments:
            - reset_param (:obj:`List`): list of reset parameters for each environment.
        Note:
            Create separate thread to reset each environment to avoid blocking.
        """
        self._check_closed()
        # clear previous info
        for env_id in self._waiting_env['step']:
            self._pipe_parents[env_id].recv()
        self._waiting_env['step'].clear()

        if reset_param is None:
            reset_param = [{} for _ in range(self.env_num)]
        self._reset_param = reset_param
        # set seed
        if hasattr(self, '_env_seed'):
            for i in range(self.env_num):
                self._pipe_parents[i].send(['seed', [self._env_seed[i]], {}])
            ret = [p.recv() for _, p in self._pipe_parents.items()]
            self._check_data(ret)

        # reset env
        reset_thread_list = []
        for env_id in range(self.env_num):
            reset_thread = PropagatingThread(target=self._reset, args=(env_id, ))
            reset_thread.daemon = True
            reset_thread_list.append(reset_thread)
        for t in reset_thread_list:
            t.start()
        for t in reset_thread_list:
            t.join()

    def _reset(self, env_id: int) -> None:

        @retry_wrapper
        def reset_fn():
            if self._pipe_parents[env_id].poll():
                recv_data = self._pipe_parents[env_id].recv()
                raise Exception("unread data left before sending to the pipe: {}".format(repr(recv_data)))
            self._pipe_parents[env_id].send(['reset', [], self._reset_param[env_id]])

            if not self._pipe_parents[env_id].poll(self.reset_timeout):  # wait for 60 seconds to restart the env
                # terminate the old subprocess
                self._pipe_parents[env_id].close()
                if self._subprocesses[env_id].is_alive():
                    self._subprocesses[env_id].terminate()
                # reset the subprocess
                self._create_env_subprocess(env_id)
                raise Exception("env reset timeout")  # leave it to retry_wrapper to try again

            obs = self._pipe_parents[env_id].recv()
            self._check_data([obs], close=False)
            if self.shared_memory:
                obs = self._obs_buffers[env_id].get()
            # Because each thread updates the corresponding env_id value, they won't lead to a thread-safe problem.
            self._env_states[env_id] = EnvState.RUN
            self._ready_obs[env_id] = self._inv_transform(obs)

        try:
            reset_fn()
        except Exception as e:
            if self._closed:  # exception cased by main thread closing parent_remote
                return
            else:
                self.close()
                raise e

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        """
        Overview:
            Wrapper of step function in the environment.
        Arguments:
            - actions (:obj:`Dict[int, Any]`): {env_id: action}
        Return:
            - timesteps (:obj:`Dict[int, namedtuple]`): {env_id: timestep}. \
                Each element of timestep is in ``torch.Tensor`` type.
        Note:
            - The env_id that appears in ``actions`` will also be returned in ``timesteps``.
            - Each environment is run by a subprocess seperately. Once an environment is done, it is reset immediately.
        Example:
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
            >>>     timesteps = env_manager.step(actions_dict):
            >>>     for env_id, timestep in timesteps.items():
            >>>         pass
        """
        self._check_closed()
        env_ids = list(actions.keys())
        assert all([self._env_states[env_id] == EnvState.RUN for env_id in env_ids]
                   ), 'current env state are: {}, please check whether the requested env is in reset or done'.format(
                       {env_id: self._env_states[env_id]
                        for env_id in env_ids}
                   )

        for env_id, act in actions.items():
            act = self._transform(act)
            self._pipe_parents[env_id].send(['step', [act], {}])

        step_args = self._async_args['step']
        wait_num, timeout = min(step_args['wait_num'], len(env_ids)), step_args['timeout']

        rest_env_ids = list(set(env_ids).union(self._waiting_env['step']))
        ready_env_ids = []
        timesteps = {}
        cur_rest_env_ids = copy.deepcopy(rest_env_ids)
        while True:
            rest_conn = [self._pipe_parents[env_id] for env_id in cur_rest_env_ids]
            ready_conn, ready_ids = AsyncSubprocessEnvManager.wait(rest_conn, min(wait_num, len(rest_conn)), timeout)
            cur_ready_env_ids = [cur_rest_env_ids[env_id] for env_id in ready_ids]
            assert len(cur_ready_env_ids) == len(ready_conn)
            timesteps.update({env_id: p.recv() for env_id, p in zip(cur_ready_env_ids, ready_conn)})
            self._check_data(timesteps.values())
            ready_env_ids += cur_ready_env_ids
            cur_rest_env_ids = list(set(cur_rest_env_ids).difference(set(cur_ready_env_ids)))
            # at least one not done timestep or all the connection is ready
            if any([not t.done for t in timesteps.values()]) or len(ready_conn) == len(rest_conn):
                break

        self._waiting_env['step']: set
        for env_id in rest_env_ids:
            if env_id in ready_env_ids:
                if env_id in self._waiting_env['step']:
                    self._waiting_env['step'].remove(env_id)
            else:
                self._waiting_env['step'].add(env_id)

        if self.shared_memory:
            for i, (env_id, timestep) in enumerate(timesteps.items()):
                timesteps[env_id] = timestep._replace(obs=self._obs_buffers[env_id].get())
        timesteps = self._inv_transform(timesteps)

        for i, (env_id, timestep) in enumerate(timesteps.items()):
            if timestep.info.get('abnormal', False):
                self._env_states[env_id] = EnvState.RESET
                reset_thread = PropagatingThread(target=self._reset, args=(env_id, ), name='abnormal_reset')
                reset_thread.daemon = True
                reset_thread.start()
                continue
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] >= self._episode_num:
                    self._env_states[env_id] = EnvState.DONE
                else:
                    self._env_states[env_id] = EnvState.RESET
                    reset_thread = PropagatingThread(target=self._reset, args=(env_id, ), name='regular_reset')
                    reset_thread.daemon = True
                    reset_thread.start()
            else:
                self._ready_obs[env_id] = timestep.obs

        return timesteps

    # This method must be staticmethod, otherwise there will be some resource conflicts(e.g. port or file)
    # Env must be created in worker, which is a trick of avoiding env pickle errors.
    @staticmethod
    def worker_fn(p: connection.Connection, c: connection.Connection, env_fn_wrapper: 'PickleWrapper', obs_buffer: ShmBuffer, method_name_list: list) -> None:  # noqa
        """
        Overview:
            Subprocess's target function to run.
        """
        torch.set_num_threads(1)
        env_fn = env_fn_wrapper.data
        env = env_fn()
        p.close()
        try:
            while True:
                try:
                    cmd, args, kwargs = c.recv()
                except EOFError:  # for the case when the pipe has been closed
                    c.close()
                    break
                try:
                    if cmd == 'getattr':
                        ret = getattr(env, args[0])
                    elif cmd in method_name_list:
                        if cmd == 'step':
                            timestep = env.step(*args, **kwargs)
                            if timestep.info.get('abnormal', False):
                                ret = timestep
                            else:
                                if obs_buffer is not None:
                                    obs_buffer.fill(timestep.obs)
                                    timestep = timestep._replace(obs=None)
                                ret = timestep
                        elif cmd == 'reset':
                            ret = env.reset(*args, **kwargs)  # obs
                            if obs_buffer is not None:
                                obs_buffer.fill(ret)
                                ret = None
                        elif args is None and kwargs is None:
                            ret = getattr(env, cmd)()
                        else:
                            ret = getattr(env, cmd)(*args, **kwargs)
                    else:
                        raise KeyError("not support env cmd: {}".format(cmd))
                    c.send(ret)
                except Exception as e:
                    # when there are some errors in env, worker_fn will send the errors to env manager
                    # directly send error to another process will lose the stack trace, so we create a new Exception
                    c.send(
                        CloudPickleWrapper(
                            e.__class__(
                                '\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)
                            )
                        )
                    )
                if cmd == 'close':
                    c.close()
                    break
        except KeyboardInterrupt:
            c.close()

    def _check_data(self, data: Iterable, close: bool = True) -> None:
        for d in data:
            if isinstance(d, Exception):
                # when receiving env Exception, env manager will safely close and raise this Exception to caller
                if close:
                    self.close()
                raise d

    # override
    def __getattr__(self, key: str) -> Any:
        self._check_closed()
        # we suppose that all the envs has the same attributes, if you need different envs, please
        # create different env managers.
        if not hasattr(self._env_ref, key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._env_ref), key))
        if isinstance(getattr(self._env_ref, key), MethodType) and key not in self.method_name_list:
            raise RuntimeError("env getattr doesn't supports method({}), please override method_name_list".format(key))
        for _, p in self._pipe_parents.items():
            p.send(['getattr', [key], {}])
        ret = [p.recv() for _, p in self._pipe_parents.items()]
        self._check_data(ret)
        return ret

    # override
    def enable_save_replay(self, replay_path: Union[List[str], str]) -> None:
        if isinstance(replay_path, str):
            replay_path = [replay_path] * self.env_num
        self._env_replay_path = replay_path

    # override
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._env_ref.close()
        for _, p in self._pipe_parents.items():
            p.send(['close', None, None])
        for _, p in self._pipe_parents.items():
            p.recv()
        # disable process join for avoiding hang
        # for p in self._subprocesses:
        #     p.join()
        for _, p in self._subprocesses.items():
            p.terminate()
        for _, p in self._pipe_parents.items():
            p.close()

    @staticmethod
    def wait(rest_conn: list, wait_num: int, timeout: Optional[float] = None) -> Tuple[list, list]:
        """
        Overview:
            Wait at least enough(len(ready_conn) >= wait_num) connections within timeout constraint.
            If timeout is None and wait_num == len(ready_conn), means sync mode;
            If timeout is not None, will return when len(ready_conn) >= wait_num and 
            this method takes more than timeout seconds.
        """
        assert 1 <= wait_num <= len(rest_conn
                                    ), 'please indicate proper wait_num: <wait_num: {}, rest_conn_num: {}>'.format(
                                        wait_num, len(rest_conn)
                                    )
        rest_conn_set = set(rest_conn)
        ready_conn = set()
        start_time = time.time()
        while len(rest_conn_set) > 0:
            if len(ready_conn) >= wait_num and timeout:
                if (time.time() - start_time) >= timeout:
                    break
            finish_conn = set(connection.wait(rest_conn_set, timeout=timeout))
            ready_conn = ready_conn.union(finish_conn)
            rest_conn_set = rest_conn_set.difference(finish_conn)
        ready_ids = [rest_conn.index(c) for c in ready_conn]
        return list(ready_conn), ready_ids


@ENV_MANAGER_REGISTRY.register('subprocess')
class SyncSubprocessEnvManager(AsyncSubprocessEnvManager):

    def _setup_async_args(self) -> None:
        self._async_args = {
            'step': {
                'wait_num': self._env_num,  # math.inf,
                'timeout': None,
            },
        }

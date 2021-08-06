from typing import Any, Union, List, Tuple, Dict, Callable, Optional
from multiprocessing import Pipe, connection, get_context, Array
from collections import namedtuple
import logging
import platform
import time
import copy
import traceback
import numpy as np
import torch
import ctypes
import pickle
import cloudpickle
from easydict import EasyDict
from types import MethodType

from ding.utils import PropagatingThread, LockContextType, LockContext, ENV_MANAGER_REGISTRY
from .base_env_manager import BaseEnvManager, EnvState, retry_wrapper, timeout_wrapper

_NTYPE_TO_CTYPE = {
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


class ShmBuffer():
    """
    Overview:
        Shared memory buffer to store numpy array.
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
        Support multiple shared memory buffers. Each key-value is name-buffer.
    """

    def __init__(self, dtype: np.generic, shape: Union[Dict[Any, tuple], tuple]) -> None:
        """
        Overview:
            Initialize the buffer container.
        Arguments:
            - dtype (:obj:`np.generic`): dtype of the data to limit the size of the buffer.
            - shape (:obj:`Union[Dict[Any, tuple], tuple]`): If `Dict[Any, tuple]`, use a dict to manage \
                multiple buffers; If `tuple`, use single buffer.
        """
        if isinstance(shape, dict):
            self._data = {k: ShmBufferContainer(dtype, v) for k, v in shape.items()}
        elif isinstance(shape, (tuple, list)):
            self._data = ShmBuffer(dtype, shape)
        else:
            raise RuntimeError("not support shape: {}".format(shape))
        self._shape = shape

    def fill(self, src_arr: Union[Dict[Any, np.ndarray], np.ndarray]) -> None:
        """
        Overview:
            Fill the one or many shared memory buffer.
        Arguments:
            - src_arr (:obj:`Union[Dict[Any, np.ndarray], np.ndarray]`): array to fill the buffer.
        """
        if isinstance(self._shape, dict):
            for k in self._shape.keys():
                self._data[k].fill(src_arr[k])
        elif isinstance(self._shape, (tuple, list)):
            self._data.fill(src_arr)

    def get(self) -> Union[Dict[Any, np.ndarray], np.ndarray]:
        """
        Overview:
            Get the one or many arrays stored in the buffer.
        Return:
            - data (:obj:`np.ndarray`): The array(s) stored in the buffer.
        """
        if isinstance(self._shape, dict):
            return {k: self._data[k].get() for k in self._shape.keys()}
        elif isinstance(self._shape, (tuple, list)):
            return self._data.get()


class CloudPickleWrapper:
    """
    Overview:
        CloudPickleWrapper can be able to pickle more python object(e.g: an object with lambda expression)
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        if isinstance(data, (tuple, list, np.ndarray)):  # pickle is faster
            self.data = pickle.loads(data)
        else:
            self.data = cloudpickle.loads(data)


@ENV_MANAGER_REGISTRY.register('async_subprocess')
class AsyncSubprocessEnvManager(BaseEnvManager):
    """
    Overview:
        Create an AsyncSubprocessEnvManager to manage multiple environments.
        Each Environment is run by a respective subprocess.
    Interfaces:
        seed, launch, ready_obs, step, reset, env_info，active_env
    """

    config = dict(
        episode_num=float("inf"),
        max_retry=5,
        step_timeout=60,
        auto_reset=True,
        reset_timeout=60,
        retry_waiting_time=0.1,
        # subprocess specified args
        shared_memory=True,
        context='spawn' if platform.system().lower() == 'windows' else 'fork',
        wait_num=2,
        step_wait_timeout=0.01,
        connect_timeout=60,
    )

    def __init__(
            self,
            env_fn: List[Callable],
            cfg: EasyDict = EasyDict({}),
    ) -> None:
        """
        Overview:
            Initialize the AsyncSubprocessEnvManager.
        Arguments:
            - env_fn (:obj:`List[Callable]`): The function to create environment
            - cfg (:obj:`EasyDict`): Config

        .. note::

            - wait_num: for each time the minimum number of env return to gather
            - step_wait_timeout: for each time the minimum number of env return to gather
        """
        super().__init__(env_fn, cfg)
        self._shared_memory = self._cfg.shared_memory
        self._context = self._cfg.context
        self._wait_num = self._cfg.wait_num
        self._step_wait_timeout = self._cfg.step_wait_timeout

        self._lock = LockContext(LockContextType.THREAD_LOCK)
        self._connect_timeout = self._cfg.connect_timeout
        self._connect_timeout = np.max([self._connect_timeout, self._step_timeout + 0.5, self._reset_timeout + 0.5])
        self._async_args = {
            'step': {
                'wait_num': min(self._wait_num, self._env_num),
                'timeout': self._step_wait_timeout
            }
        }

    def _create_state(self) -> None:
        r"""
        Overview:
            Fork/spawn sub-processes(Call ``_create_env_subprocess``) and create pipes to transfer the data.
        """
        self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
        self._ready_obs = {env_id: None for env_id in range(self.env_num)}
        self._env_ref = self._env_fn[0]()
        self._reset_param = {i: {} for i in range(self.env_num)}
        if self._shared_memory:
            obs_space = self._env_ref.info().obs_space
            shape = obs_space.shape
            dtype = np.dtype(obs_space.value['dtype']) if obs_space.value is not None else np.dtype(np.float32)
            self._obs_buffers = {env_id: ShmBufferContainer(dtype, shape) for env_id in range(self.env_num)}
        else:
            self._obs_buffers = {env_id: None for env_id in range(self.env_num)}
        self._pipe_parents, self._pipe_children = {}, {}
        self._subprocesses = {}
        for env_id in range(self.env_num):
            self._create_env_subprocess(env_id)
        self._waiting_env = {'step': set()}
        self._closed = False

    def _create_env_subprocess(self, env_id):
        # start a new one
        self._pipe_parents[env_id], self._pipe_children[env_id] = Pipe()
        ctx = get_context(self._context)
        self._subprocesses[env_id] = ctx.Process(
            # target=self.worker_fn,
            target=self.worker_fn_robust,
            args=(
                self._pipe_parents[env_id],
                self._pipe_children[env_id],
                CloudPickleWrapper(self._env_fn[env_id]),
                self._obs_buffers[env_id],
                self.method_name_list,
                self._reset_timeout,
                self._step_timeout,
                self._max_retry,
            ),
            daemon=True,
            name='subprocess_env_manager{}_{}'.format(env_id, time.time())
        )
        self._subprocesses[env_id].start()
        self._pipe_children[env_id].close()
        self._env_states[env_id] = EnvState.INIT

        if self._env_replay_path is not None:
            self._pipe_parents[env_id].send(['enable_save_replay', [self._env_replay_path[env_id]], {}])
            self._pipe_parents[env_id].recv()

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
            The observations are returned in np.ndarray.
        Example:
            >>>     obs_dict = env_manager.ready_obs
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
        """
        no_done_env_idx = [i for i, s in self._env_states.items() if s != EnvState.DONE]
        sleep_count = 0
        while not any([self._env_states[i] == EnvState.RUN for i in no_done_env_idx]):
            if sleep_count % 1000 == 0:
                logging.warning(
                    'VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count)
                )
            time.sleep(0.001)
            sleep_count += 1
        return {i: self._ready_obs[i] for i in self.ready_env}

    def launch(self, reset_param: Optional[Dict] = None) -> None:
        """
        Overview:
            Set up the environments and their parameters.
        Arguments:
            - reset_param (:obj:`Optional[Dict]`): Dict of reset parameters for each environment, key is the env_id, \
                value is the cooresponding reset parameters.
        """
        assert self._closed, "please first close the env manager"
        if reset_param is not None:
            assert len(reset_param) == len(self._env_fn)
        self._create_state()
        self.reset(reset_param)

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        """
        Overview:
            Reset the environments their parameters.
        Arguments:
            - reset_param (:obj:`List`): Dict of reset parameters for each environment, key is the env_id, \
                value is the cooresponding reset parameters.
        """
        self._check_closed()
        # clear previous info
        for env_id in self._waiting_env['step']:
            self._pipe_parents[env_id].recv()
        self._waiting_env['step'].clear()

        if reset_param is None:
            reset_env_list = [env_id for env_id in range(self._env_num)]
        else:
            reset_env_list = reset_param.keys()
            for env_id in reset_param:
                self._reset_param[env_id] = reset_param[env_id]

        sleep_count = 0
        while any([self._env_states[i] == EnvState.RESET for i in reset_env_list]):
            if sleep_count % 1000 == 0:
                logging.warning(
                    'VEC_ENV_MANAGER: not all the envs finish resetting, sleep {} times'.format(sleep_count)
                )
            time.sleep(0.001)
            sleep_count += 1

        # reset env
        reset_thread_list = []
        for i, env_id in enumerate(reset_env_list):
            self._env_states[env_id] = EnvState.RESET
            # set seed
            if self._env_seed[env_id] is not None:
                try:
                    if self._env_dynamic_seed is not None:
                        self._pipe_parents[env_id].send(['seed', [self._env_seed[env_id], self._env_dynamic_seed], {}])
                    else:
                        self._pipe_parents[env_id].send(['seed', [self._env_seed[env_id]], {}])
                    ret = self._pipe_parents[env_id].recv()
                    self._check_data({env_id: ret})
                    self._env_seed[env_id] = None  # seed only use once
                except Exception as e:
                    logging.warning("subprocess reset set seed failed, ignore and continue...")
            reset_thread = PropagatingThread(target=self._reset, args=(env_id, ))
            reset_thread.daemon = True
            reset_thread_list.append(reset_thread)

        for t in reset_thread_list:
            t.start()
        for t in reset_thread_list:
            t.join()

    def _reset(self, env_id: int) -> None:

        @retry_wrapper(max_retry=self._max_retry, waiting_time=self._retry_waiting_time)
        def reset_fn():
            if self._pipe_parents[env_id].poll():
                recv_data = self._pipe_parents[env_id].recv()
                raise Exception("unread data left before sending to the pipe: {}".format(repr(recv_data)))
            # if self._reset_param[env_id] is None, just reset specific env, not pass reset param
            if self._reset_param[env_id] is not None:
                assert isinstance(self._reset_param[env_id], dict), type(self._reset_param[env_id])
                self._pipe_parents[env_id].send(['reset', [], self._reset_param[env_id]])
            else:
                self._pipe_parents[env_id].send(['reset', [], {}])

            if not self._pipe_parents[env_id].poll(self._connect_timeout):
                # terminate the old subprocess
                self._pipe_parents[env_id].close()
                if self._subprocesses[env_id].is_alive():
                    self._subprocesses[env_id].terminate()
                # reset the subprocess
                self._create_env_subprocess(env_id)
                raise Exception("env reset timeout")  # Leave it to retry_wrapper to try again

            obs = self._pipe_parents[env_id].recv()
            self._check_data({env_id: obs}, close=False)
            if self._shared_memory:
                obs = self._obs_buffers[env_id].get()
            # Because each thread updates the corresponding env_id value, they won't lead to a thread-safe problem.
            self._env_states[env_id] = EnvState.RUN
            self._ready_obs[env_id] = obs

        try:
            reset_fn()
        except Exception as e:
            logging.error('VEC_ENV_MANAGER: env {} reset error'.format(env_id))
            logging.error('\nEnv Process Reset Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
            if self._closed:  # exception cased by main thread closing parent_remote
                return
            else:
                self.close()
                raise e

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        """
        Overview:
            Step all environments. Reset an env if done.
        Arguments:
            - actions (:obj:`Dict[int, Any]`): {env_id: action}
        Returns:
            - timesteps (:obj:`Dict[int, namedtuple]`): {env_id: timestep}. Timestep is a \
                ``BaseEnvTimestep`` tuple with observation, reward, done, env_info.
        Example:
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
            >>>     timesteps = env_manager.step(actions_dict):
            >>>     for env_id, timestep in timesteps.items():
            >>>         pass

        .. note:

            - The env_id that appears in ``actions`` will also be returned in ``timesteps``.
            - Each environment is run by a subprocess separately. Once an environment is done, it is reset immediately.
            - Async subprocess env manager use ``connection.wait`` to poll.
        """
        self._check_closed()
        env_ids = list(actions.keys())
        assert all([self._env_states[env_id] == EnvState.RUN for env_id in env_ids]
                   ), 'current env state are: {}, please check whether the requested env is in reset or done'.format(
                       {env_id: self._env_states[env_id]
                        for env_id in env_ids}
                   )

        for env_id, act in actions.items():
            self._pipe_parents[env_id].send(['step', [act], {}])

        timesteps = {}
        step_args = self._async_args['step']
        wait_num, timeout = min(step_args['wait_num'], len(env_ids)), step_args['timeout']
        rest_env_ids = list(set(env_ids).union(self._waiting_env['step']))
        ready_env_ids = []
        cur_rest_env_ids = copy.deepcopy(rest_env_ids)
        while True:
            rest_conn = [self._pipe_parents[env_id] for env_id in cur_rest_env_ids]
            ready_conn, ready_ids = AsyncSubprocessEnvManager.wait(rest_conn, min(wait_num, len(rest_conn)), timeout)
            cur_ready_env_ids = [cur_rest_env_ids[env_id] for env_id in ready_ids]
            assert len(cur_ready_env_ids) == len(ready_conn)
            timesteps.update({env_id: p.recv() for env_id, p in zip(cur_ready_env_ids, ready_conn)})
            self._check_data(timesteps)
            ready_env_ids += cur_ready_env_ids
            cur_rest_env_ids = list(set(cur_rest_env_ids).difference(set(cur_ready_env_ids)))
            # At least one not done env timestep, or all envs' steps are finished
            if any([not t.done for t in timesteps.values()]) or len(ready_conn) == len(rest_conn):
                break
        self._waiting_env['step']: set
        for env_id in rest_env_ids:
            if env_id in ready_env_ids:
                if env_id in self._waiting_env['step']:
                    self._waiting_env['step'].remove(env_id)
            else:
                self._waiting_env['step'].add(env_id)

        if self._shared_memory:
            for i, (env_id, timestep) in enumerate(timesteps.items()):
                timesteps[env_id] = timestep._replace(obs=self._obs_buffers[env_id].get())

        for env_id, timestep in timesteps.items():
            if timestep.info.get('abnormal', False):
                self._env_states[env_id] = EnvState.ERROR
                continue
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num and self._auto_reset:
                    self._env_states[env_id] = EnvState.RESET
                    reset_thread = PropagatingThread(target=self._reset, args=(env_id, ), name='regular_reset')
                    reset_thread.daemon = True
                    reset_thread.start()
                else:
                    self._env_states[env_id] = EnvState.DONE
            else:
                self._ready_obs[env_id] = timestep.obs
        return timesteps

    # This method must be staticmethod, otherwise there will be some resource conflicts(e.g. port or file)
    # Env must be created in worker, which is a trick of avoiding env pickle errors.
    # A more robust version is used by default. But this one is also preserved.
    @staticmethod
    def worker_fn(
            p: connection.Connection, c: connection.Connection, env_fn_wrapper: 'CloudPickleWrapper',
            obs_buffer: ShmBuffer, method_name_list: list
    ) -> None:  # noqa
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
                        e.__class__(
                            '\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)
                        )
                    )
                if cmd == 'close':
                    c.close()
                    break
        except KeyboardInterrupt:
            c.close()

    @staticmethod
    def worker_fn_robust(
            parent,
            child,
            env_fn_wrapper,
            obs_buffer,
            method_name_list,
            reset_timeout=60,
            step_timeout=60,
            max_retry=1
    ) -> None:
        """
        Overview:
            A more robust version of subprocess's target function to run. Used by default.
        """
        torch.set_num_threads(1)
        env_fn = env_fn_wrapper.data
        env = env_fn()
        parent.close()

        @retry_wrapper(max_retry=max_retry)
        @timeout_wrapper(timeout=step_timeout)
        def step_fn(*args, **kwargs):
            timestep = env.step(*args, **kwargs)
            if timestep.info.get('abnormal', False):
                ret = timestep
            else:
                if obs_buffer is not None:
                    obs_buffer.fill(timestep.obs)
                    timestep = timestep._replace(obs=None)
                ret = timestep
            return ret

        # self._reset method has add retry_wrapper decorator
        @timeout_wrapper(timeout=reset_timeout)
        def reset_fn(*args, **kwargs):
            try:
                ret = env.reset(*args, **kwargs)
                if obs_buffer is not None:
                    obs_buffer.fill(ret)
                    ret = None
                return ret
            except Exception as e:
                env.close()
                raise e

        while True:
            try:
                cmd, args, kwargs = child.recv()
            except EOFError:  # for the case when the pipe has been closed
                child.close()
                break
            try:
                if cmd == 'getattr':
                    ret = getattr(env, args[0])
                elif cmd in method_name_list:
                    if cmd == 'step':
                        ret = step_fn(*args, **kwargs)
                    elif cmd == 'reset':
                        ret = reset_fn(*args, **kwargs)
                    elif args is None and kwargs is None:
                        ret = getattr(env, cmd)()
                    else:
                        ret = getattr(env, cmd)(*args, **kwargs)
                else:
                    raise KeyError("not support env cmd: {}".format(cmd))
                child.send(ret)
            except Exception as e:
                # print("Sub env '{}' error when executing {}".format(str(env), cmd))
                # when there are some errors in env, worker_fn will send the errors to env manager
                # directly send error to another process will lose the stack trace, so we create a new Exception
                child.send(
                    e.__class__('\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
                )
            if cmd == 'close':
                child.close()
                break

    def _check_data(self, data: Dict, close: bool = True) -> None:
        exceptions = []
        for i, d in data.items():
            if isinstance(d, Exception):
                self._env_states[i] = EnvState.ERROR
                exceptions.append(d)
        # when receiving env Exception, env manager will safely close and raise this Exception to caller
        if len(exceptions) > 0:
            if close:
                self.close()
            raise exceptions[0]

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
        data = {i: p.recv() for i, p in self._pipe_parents.items()}
        self._check_data(data)
        ret = [data[i] for i in self._pipe_parents.keys()]
        return ret

    # override
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

    # override
    def close(self) -> None:
        """
        Overview:
            CLose the env manager and release all related resources.
        """
        if self._closed:
            return
        self._closed = True
        self._env_ref.close()
        for _, p in self._pipe_parents.items():
            p.send(['close', None, None])
        for _, p in self._pipe_parents.items():
            p.recv()
        for i in range(self._env_num):
            self._env_states[i] = EnvState.VOID
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
    config = dict(
        episode_num=float("inf"),
        max_retry=5,
        step_timeout=60,
        auto_reset=True,
        reset_timeout=60,
        retry_waiting_time=0.1,
        # subprocess specified args
        shared_memory=True,
        context='spawn' if platform.system().lower() == 'windows' else 'fork',
        wait_num=float("inf"),  # inf mean all the environments
        step_wait_timeout=None,
        connect_timeout=60,
        force_reproducibility=False,
    )

    def __init__(
            self,
            env_fn: List[Callable],
            cfg: EasyDict = EasyDict({}),
    ) -> None:
        super(SyncSubprocessEnvManager, self).__init__(env_fn, cfg)
        self._force_reproducibility = self._cfg.force_reproducibility

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        """
        Overview:
            Step all environments. Reset an env if done.
        Arguments:
            - actions (:obj:`Dict[int, Any]`): {env_id: action}
        Returns:
            - timesteps (:obj:`Dict[int, namedtuple]`): {env_id: timestep}. Timestep is a \
                ``BaseEnvTimestep`` tuple with observation, reward, done, env_info.
        Example:
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
            >>>     timesteps = env_manager.step(actions_dict):
            >>>     for env_id, timestep in timesteps.items():
            >>>         pass

        .. note::

            - The env_id that appears in ``actions`` will also be returned in ``timesteps``.
            - Each environment is run by a subprocess separately. Once an environment is done, it is reset immediately.
        """
        self._check_closed()
        env_ids = list(actions.keys())
        assert all([self._env_states[env_id] == EnvState.RUN for env_id in env_ids]
                   ), 'current env state are: {}, please check whether the requested env is in reset or done'.format(
                       {env_id: self._env_states[env_id]
                        for env_id in env_ids}
                   )
        for env_id, act in actions.items():
            self._pipe_parents[env_id].send(['step', [act], {}])

        # ===     This part is different from async one.     ===
        # === Because operate in this way is more efficient. ===
        timesteps = {}
        ready_conn = [self._pipe_parents[env_id] for env_id in env_ids]
        timesteps.update({env_id: p.recv() for env_id, p in zip(env_ids, ready_conn)})
        self._check_data(timesteps)
        # ======================================================

        if self._shared_memory:
            for i, (env_id, timestep) in enumerate(timesteps.items()):
                timesteps[env_id] = timestep._replace(obs=self._obs_buffers[env_id].get())
        for env_id, timestep in timesteps.items():
            if timestep.info.get('abnormal', False):
                self._env_states[env_id] = EnvState.ERROR
                continue
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num and self._auto_reset:
                    self._env_states[env_id] = EnvState.RESET
                    reset_thread = PropagatingThread(target=self._reset, args=(env_id, ), name='regular_reset')
                    reset_thread.daemon = True
                    reset_thread.start()
                    if self._force_reproducibility:
                        reset_thread.join()
                else:
                    self._env_states[env_id] = EnvState.DONE
            else:
                self._ready_obs[env_id] = timestep.obs
        return timesteps

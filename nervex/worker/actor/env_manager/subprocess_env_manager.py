from multiprocessing import Process, Pipe, connection, get_context, Array
from collections import namedtuple
import enum
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
from nervex.utils import PropagatingThread, LockContextType, LockContext
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
            - shape (:obj:`Tuple`): shape of the data to limit the size of the buffer.
        """
        self.buffer = Array(_NTYPE_TO_CTYPE[dtype.type], int(np.prod(shape)))
        self.dtype = dtype
        self.shape = shape

    def fill(self, src_arr: np.ndarray) -> None:
        """
        Overview:
            Fill the shared memory buffer with a numpy array.
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
            A copy of the data stored in the buffer.
        """
        arr = np.frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)
        return arr.copy()


class ShmBufferContainer(object):
    """
    Overview:
        Support dictionary of shared memory buffer.
    """

    def __init__(self, dtype: np.generic, shape: Union[dict, tuple]) -> None:
        if isinstance(shape, dict):
            self._data = {k: ShmBufferContainer(dtype, v) for k, v in shape.items()}
        elif isinstance(shape, (tuple, list)):
            self._data = ShmBuffer(dtype, shape)
        else:
            raise RuntimeError("not support shape: {}".format(shape))
        self._shape = shape

    def fill(self, src_arr: Union[dict, np.ndarray]) -> None:
        if isinstance(self._shape, dict):
            for k in self._shape.keys():
                self._data[k].fill(src_arr[k])
        elif isinstance(self._shape, (tuple, list)):
            self._data.fill(src_arr)

    def get(self) -> Union[dict, np.ndarray]:
        if isinstance(self._shape, dict):
            return {k: self._data[k].get() for k in self._shape.keys()}
        elif isinstance(self._shape, (tuple, list)):
            return self._data.get()


class CloudpickleWrapper(object):
    """
    Overview:
        CloudpickleWrapper can be able to pickle more python object(e.g: an object with lambda expression)
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> bytes:
        if isinstance(self.data, (tuple, list, np.ndarray)):  # pickle is faster
            return pickle.dumps(self.data)
        else:
            return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        if isinstance(data, (tuple, list, np.ndarray)):  # pickle is faster
            self.data = pickle.loads(data)
        else:
            self.data = cloudpickle.loads(data)


def retry_wrapper(fn: Callable, max_retry: int = 10) -> Callable:
    """
    Overview:
        Retry the function until exceeding the maximum times.
    """

    def wrapper(*args, **kwargs):
        exceptions = []
        for _ in range(max_retry):
            try:
                ret = fn(*args, **kwargs)
                return ret
            except Exception as e:
                exceptions.append(e)
                time.sleep(0.5)
        e_info = ''.join(
            [
                'Retry {} failed from:\n {}\n'.format(i, ''.join(traceback.format_tb(e.__traceback__)) + str(e))
                for i, e in enumerate(exceptions)
            ]
        )
        fn_exception = Exception("Function {} runtime error:\n{}".format(fn, e_info))
        raise RuntimeError("Function {} has exceeded max retries({})".format(fn, max_retry)) from fn_exception

    return wrapper


class SubprocessEnvManager(BaseEnvManager):
    """
    Overview:
        Create a SubprocessEnvManager to manage multiple environments. Each Environment is run by a seperate subprocess.

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
            Initialize the SubprocessEnvManager.
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
        self._lock = LockContext(LockContextType.THREAD_LOCK)

    def _create_state(self) -> None:
        r"""
        Overview:
            Fork/spawn sub-processes and create pipes to convey the data.
        """
        self._closed = False
        self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
        self._env_done = {env_id: False for env_id in range(self.env_num)}
        self._next_obs = {env_id: None for env_id in range(self.env_num)}
        if self.shared_memory:
            obs_space = self._env_ref.info().obs_space
            shape = obs_space.shape
            dtype = np.dtype(obs_space.value['dtype']) if obs_space.value is not None else np.dtype(np.float32)
            self._obs_buffers = {env_id: ShmBufferContainer(dtype, shape) for env_id in range(self.env_num)}
        else:
            self._obs_buffers = {env_id: None for env_id in range(self.env_num)}
        self._parent_remote, self._child_remote = zip(*[Pipe() for _ in range(self.env_num)])
        ctx = get_context(self.context_str)
        # due to the runtime delay of lambda expression, we use partial for the generation of different envs,
        # otherwise, it will only use the last item cfg.
        env_fn = [partial(self._env_fn, cfg=self._env_cfg[env_id]) for env_id in range(self.env_num)]
        self._processes = [
            ctx.Process(
                target=self.worker_fn,
                args=(parent, child, CloudpickleWrapper(fn), obs_buffer, self.method_name_list),
                daemon=True,
                name='subprocess_env_manager{}_{}'.format(idx, time.time())
            )
            for idx, (parent, child, fn, obs_buffer
                      ) in enumerate(zip(self._parent_remote, self._child_remote, env_fn, self._obs_buffers.values()))
        ]
        for p in self._processes:
            p.start()
        for c in self._child_remote:
            c.close()
        self._env_state = {env_id: EnvState.INIT for env_id in range(self.env_num)}
        self._waiting_env = {'step': set()}
        self._setup_async_args()
        if hasattr(self, '_env_replay_path'):
            for p, s in zip(self._parent_remote, self._env_replay_path):
                p.send(CloudpickleWrapper(['enable_save_replay', [s], {}]))
            for p in self._parent_remote:
                p.recv()

    def _setup_async_args(self) -> None:
        r"""
        Overview:
            set up the async arguments utilized in the step().
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
        return [i for i, s in self._env_state.items() if s == EnvState.RUN]

    @property
    def ready_env(self) -> List[int]:
        return [i for i in self.active_env if i not in self._waiting_env['step']]

    @property
    def next_obs(self) -> Dict[int, Any]:
        """
        Overview:
            Get the next observations.
        Return:
            A dictionary with observations and their environment IDs.
        Note:
            The observations are returned in torch.Tensor.
        Example:
            >>>     obs_dict = env_manager.next_obs
            >>>     action_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
        """
        no_done_env_idx = [i for i, s in self._env_state.items() if s != EnvState.DONE]
        sleep_count = 0
        while all([self._env_state[i] == EnvState.RESET for i in no_done_env_idx]):
            print('VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count))
            time.sleep(1)
            sleep_count += 1
        return self._inv_transform({i: self._next_obs[i] for i in self.ready_env})

    @property
    def done(self) -> bool:
        return all([s == EnvState.DONE for s in self._env_state.values()])

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
            self._parent_remote[env_id].recv()
        self._waiting_env['step'].clear()

        if reset_param is None:
            reset_param = [{} for _ in range(self.env_num)]
        self._reset_param = reset_param
        # set seed
        if hasattr(self, '_env_seed'):
            for i in range(self.env_num):
                self._parent_remote[i].send(CloudpickleWrapper(['seed', [self._env_seed[i]], {}]))
            ret = [p.recv().data for p in self._parent_remote]
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
            self._parent_remote[env_id].send(CloudpickleWrapper(['reset', [], self._reset_param[env_id]]))
            obs = self._parent_remote[env_id].recv().data
            self._check_data([obs], close=False)
            if self.shared_memory:
                obs = self._obs_buffers[env_id].get()
            with self._lock:
                self._env_state[env_id] = EnvState.RUN
                self._next_obs[env_id] = obs

        try:
            reset_fn()
        except Exception as e:
            if self._closed:  # exception cased by main thread closing parent_remote
                return
            else:
                self.close()
                raise e

    def step(self, action: Dict[int, Any]) -> Dict[int, namedtuple]:
        """
        Overview:
            Wrapper of step function in the environment.
        Arguments:
            - action (:obj:`Dict`): a dictionary, {env_id: action}, which includes actions and their env ids.
        Return:
            - timesteps (:obj:`Dict`): a dictionary, {env_id: timestep}, which includes each env's timestep.
        Note:
            - The env_id that appears in action will also be returned in timesteps.
            - Each environment is run by a subprocess seperately. Once an environment is done, it is reset immediately.
        Example:
            >>>     action_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
            >>>     timesteps = env_manager.step(action_dict):
            >>>     for env_id, timestep in timesteps.items():
            >>>         pass
        """
        self._check_closed()
        env_ids = list(action.keys())
        assert all([self._env_state[env_id] == EnvState.RUN for env_id in env_ids]
                   ), 'current env state are: {}, please check whether the requested env is in reset or done'.format(
                       {env_id: self._env_state[env_id]
                        for env_id in env_ids}
                   )

        for env_id, act in action.items():
            act = self._transform(act)
            self._parent_remote[env_id].send(CloudpickleWrapper(['step', [act], {}]))

        handle = self._async_args['step']
        wait_num, timeout = min(handle['wait_num'], len(env_ids)), handle['timeout']
        rest_env_ids = list(set(env_ids).union(self._waiting_env['step']))

        ready_env_ids = []
        timesteps = {}
        cur_rest_env_ids = copy.deepcopy(rest_env_ids)
        while True:
            rest_conn = [self._parent_remote[env_id] for env_id in cur_rest_env_ids]
            ready_conn, ready_ids = SubprocessEnvManager.wait(rest_conn, min(wait_num, len(rest_conn)), timeout)
            cur_ready_env_ids = [cur_rest_env_ids[env_id] for env_id in ready_ids]
            assert len(cur_ready_env_ids) == len(ready_conn)
            timesteps.update({env_id: p.recv().data for env_id, p in zip(cur_ready_env_ids, ready_conn)})
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

        for env_id, timestep in timesteps.items():
            if timestep.info.get('abnormal', False):
                self._env_state[env_id] = EnvState.RESET
                reset_thread = PropagatingThread(target=self._reset, args=(env_id, ), name='abnormal_reset')
                reset_thread.daemon = True
                reset_thread.start()
                continue
            if self.shared_memory:
                timestep = timestep._replace(obs=self._obs_buffers[env_id].get())
            timesteps[env_id] = timestep
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] >= self._epsiode_num:
                    self._env_state[env_id] = EnvState.DONE
                else:
                    self._env_state[env_id] = EnvState.RESET
                    reset_thread = PropagatingThread(target=self._reset, args=(env_id, ), name='regular_reset')
                    reset_thread.daemon = True
                    reset_thread.start()
            else:
                self._next_obs[env_id] = timestep.obs

        return self._inv_transform(timesteps)

    # this method must be staticmethod, otherwise there will be some resource conflicts(e.g. port or file)
    # env must be created in worker, which is a trick of avoiding env pickle errors.
    @staticmethod
    def worker_fn(p, c, env_fn_wrapper, obs_buffer, method_name_list) -> None:
        """
        Overview:
            Subprocess's target function to run.
        """
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
                    c.send(CloudpickleWrapper(ret))
                except Exception as e:
                    # when there are some errors in env, worker_fn will send the errors to env manager
                    # directly send error to another process will lose the stack trace, so we create a new Exception
                    c.send(
                        CloudpickleWrapper(
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
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['getattr', [key], {}]))
        ret = [p.recv().data for p in self._parent_remote]
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
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['close', None, None]))
        for p in self._processes:
            p.join()
        for p in self._processes:
            p.terminate()
        for p in self._parent_remote:
            p.close()

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
            finish_conn = set(connection.wait(rest_conn_set, timeout=timeout))
            ready_conn = ready_conn.union(finish_conn)
            rest_conn_set = rest_conn_set.difference(finish_conn)
            if len(ready_conn) >= wait_num and timeout:
                rest_time = timeout - (time.time() - start_time)
                if rest_time <= 0.0:
                    break
        ready_ids = [rest_conn.index(c) for c in ready_conn]
        return list(ready_conn), ready_ids


class SyncSubprocessEnvManager(SubprocessEnvManager):

    def _setup_async_args(self) -> None:
        self._async_args = {
            'step': {
                'wait_num': math.inf,
                'timeout': None,
            },
        }

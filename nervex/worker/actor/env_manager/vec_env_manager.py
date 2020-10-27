from multiprocessing import Process, Pipe
import traceback
from types import MethodType
from typing import Any, Union, List

import cloudpickle

from .base_env_manager import BaseEnvManager


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
                args=(parent, child, CloudpickleWrapper(env), self.method_name_list),
                daemon=True
            ) for parent, child, env in zip(self._parent_remote, self._child_remote, self._envs)
        ]
        for p in self._processes:
            p.start()
        for c in self._child_remote:
            c.close()

    @property
    def method_name_list(self) -> list:
        return ['reset', 'step', 'seed', 'close']

    # this method must be staticmethod, otherwise there will be some resource conflicts(e.g. port or file)
    @staticmethod
    def worker_fn(p, c, env_wrapper, method_name_list) -> None:
        env = env_wrapper.data
        p.close()
        try:
            while True:
                try:
                    cmd, data = c.recv().data
                except EOFError:  # for the case when the pipe has been closed
                    c.close()
                    break
                try:
                    if cmd == 'getattr':
                        ret = getattr(env, data)
                    elif cmd in method_name_list:
                        if data is None:
                            ret = getattr(env, cmd)()
                        else:
                            ret = getattr(env, cmd)(**data)
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

    # override
    def _execute_by_envid(
            self,
            fn_name: str,
            param: Union[None, List[dict]] = None,
            env_id: Union[None, List[int]] = None
    ) -> Union[list, dict]:
        real_env_id = list(range(self.env_num)) if env_id is None else env_id
        for i in range(len(real_env_id)):
            if param is None:
                self._parent_remote[real_env_id[i]].send(CloudpickleWrapper([fn_name, None]))
            else:
                self._parent_remote[real_env_id[i]].send(CloudpickleWrapper([fn_name, param[i]]))
        ret = {i: self.safe_recv(self._parent_remote[i]) for i in real_env_id}
        ret = list(ret.values()) if env_id is None else ret
        return ret

    def safe_recv(self, p, close=False):
        data = p.recv().data
        if isinstance(data, Exception):
            # when receiving env Exception, env manager will safely close and raise this Exception to caller
            if not close:
                self.close()
                raise data
        return data

    # override
    def __getattr__(self, key: str) -> Any:
        self._check_closed()
        # we suppose that all the envs has the same attributes, if you need different envs, please
        # create different env managers.
        if not hasattr(self._envs[0], key):
            raise AttributeError("env `{}` doesn't have the attribute `{}`".format(type(self._envs[0]), key))
        if isinstance(getattr(self._envs[0], key), MethodType):
            raise TypeError("env manager getattr doesn't supports method, please override method_name_list")
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['getattr', key]))
        return [self.safe_recv(p) for p in self._parent_remote]

    # override
    def close(self) -> None:
        if self._closed:
            return
        super().close()
        for p in self._parent_remote:
            p.send(CloudpickleWrapper(['close', None]))
        result = [self.safe_recv(p, close=True) for p in self._parent_remote]
        for p in self._processes:
            p.join()

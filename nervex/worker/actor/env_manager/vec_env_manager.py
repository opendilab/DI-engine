from multiprocessing import Process, Pipe
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
            Process(target=self.worker_fn, args=(parent, child, CloudpickleWrapper(env)), daemon=True)
            for parent, child, env in zip(self._parent_remote, self._child_remote, self._envs)
        ]
        for p in self._processes:
            p.start()
        for c in self._child_remote:
            c.close()
        self._closed = False

    @staticmethod
    def worker_fn(p, c, env_wrapper) -> None:
        env = env_wrapper.data
        p.close()
        try:
            while True:
                try:
                    cmd, data = c.recv()
                except EOFError:  # for the case when the pipe has been closed
                    c.close()
                    break
                if cmd == 'getattr':
                    c.send(getattr(env, data) if hasattr(env, data) else None)
                elif cmd in ['reset', 'step', 'seed', 'close']:
                    if data is None:
                        c.send(getattr(env, cmd)())
                    else:
                        c.send(getattr(env, cmd)(**data))
                    if cmd == 'close':
                        c.close()
                        break
                else:
                    c.close()
                    raise KeyError("not support env cmd: {}".format(cmd))
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
                self._parent_remote[real_env_id[i]].send([fn_name, None])
            else:
                self._parent_remote[real_env_id[i]].send([fn_name, param[i]])
        ret = {i: self._parent_remote[i].recv() for i in real_env_id}
        ret = list(ret.values()) if env_id is None else ret
        return ret

    # override
    def __getattr__(self, key: str) -> Any:
        for p in self._parent_remote:
            p.send(['getattr', key])
        return [p.recv() for p in self._parent_remote]

    # override
    def close(self) -> None:
        if self._closed:
            return
        super().close()
        for p in self._parent_remote:
            p.send(['close', None])
        result = [p.recv() for p in self._parent_remote]
        for p in self._processes:
            p.join()
        self._closed = True

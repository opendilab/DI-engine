"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""
import os
import socket
import time
import uuid
from typing import Optional, Any
from threading import Thread
from contextlib import closing


def get_ip() -> str:
    r"""
    Overview:
        get the ip(host) of socket
    Returns:
        - ip(:obj:`str`): the corresponding ip
    """
    # beware: return 127.0.0.1 on some slurm nodes
    myname = socket.getfqdn(socket.gethostname())
    myaddr = socket.gethostbyname(myname)

    return myaddr


def get_pid() -> int:
    r"""
    Overview:
        os.getpid
    """
    return os.getpid()


def get_task_uid() -> str:
    r"""
    Overview:
        get the slurm job_id, pid and uid
    """
    return os.getenv('SLURM_JOB_ID', 'PID{pid}UUID{uuid}'.format(
        pid=str(get_pid()),
        uuid=str(uuid.uuid1()),
    )) + '_' + str(time.time())


class PropagatingThread(Thread):
    """
    Overview:
        Subclass of Thread that propagates execution exception in the thread to the caller
    Examples:
        >>> def func():
        >>>     raise Exception()
        >>> t = PropagatingThread(target=func, args=())
        >>> t.start()
        >>> t.join()
    """

    def run(self) -> None:
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self) -> Any:
        super(PropagatingThread, self).join()
        if self.exc:
            raise RuntimeError('Exception in thread({})'.format(id(self))) from self.exc
        return self.ret


def find_free_port(host: str) -> int:
    r"""
    Overview:
        Look up the free port list and return one
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

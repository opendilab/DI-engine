import os
import socket
import time
import uuid
from contextlib import closing
from threading import Thread
from typing import Any


def get_ip() -> str:
    r"""
    Overview:
        Get the ``ip(host)`` of socket
    Returns:
        - ip(:obj:`str`): The corresponding ip
    """
    # beware: return 127.0.0.1 on some slurm nodes
    myname = socket.getfqdn(socket.gethostname())
    myaddr = socket.gethostbyname(myname)

    return myaddr


def get_pid() -> int:
    r"""
    Overview:
        ``os.getpid``
    """
    return os.getpid()


def get_task_uid() -> str:
    r"""
    Overview:
        Get the slurm ``job_id``, ``pid`` and ``uid``
    """
    return '{}_{}'.format(str(uuid.uuid4()), str(time.time())[-6:])


class PropagatingThread(Thread):
    """
    Overview:
        Subclass of Thread that propagates execution exception in the thread to the caller
    Interface:
        ``run``, ``join``
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
        except Exception as e:
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

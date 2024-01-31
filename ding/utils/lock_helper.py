import os
import multiprocessing
import threading
import platform
from enum import Enum, unique

from pathlib import Path
if platform.system().lower() != 'windows':
    import fcntl
else:
    fcntl = None


@unique
class LockContextType(Enum):
    """
    Overview:
        Enum to express the type of the lock.
    """
    THREAD_LOCK = 1
    PROCESS_LOCK = 2


_LOCK_TYPE_MAPPING = {
    LockContextType.THREAD_LOCK: threading.Lock,
    LockContextType.PROCESS_LOCK: multiprocessing.Lock,
}


class LockContext(object):
    """
    Overview:
        Generate a LockContext in order to make sure the thread safety.

    Interfaces:
        ``__init__``, ``__enter__``, ``__exit__``.

    Example:
        >>> with LockContext() as lock:
        >>>     print("Do something here.")
    """

    def __init__(self, lock_type: LockContextType = LockContextType.THREAD_LOCK):
        """
        Overview:
            Init the lock according to the given type.

        Arguments:
           - lock_type (:obj:`LockContextType`): The type of lock to be used. Defaults to LockContextType.THREAD_LOCK.
        """
        self.lock = _LOCK_TYPE_MAPPING[lock_type]()

    def acquire(self):
        """
        Overview:
            Acquires the lock.
        """
        self.lock.acquire()

    def release(self):
        """
        Overview:
            Releases the lock.
        """
        self.lock.release()

    def __enter__(self):
        """
        Overview:
            Enters the context and acquires the lock.
        """
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        """
        Overview:
            Exits the context and releases the lock.
        Arguments:
            - args (:obj:`Tuple`): The arguments passed to the ``__exit__`` function.
            - kwargs (:obj:`Dict`): The keyword arguments passed to the ``__exit__`` function.
        """
        self.lock.release()


rw_lock_mapping = {}


def get_rw_file_lock(name: str, op: str):
    """
    Overview:
        Get generated file lock with name and operator
    Arguments:
        - name (:obj:`str`): Lock's name.
        - op (:obj:`str`): Assigned operator, i.e. ``read`` or ``write``.
    Returns:
        - (:obj:`RWLockFairD`): Generated rwlock
    """
    assert op in ['read', 'write']
    try:
        from readerwriterlock import rwlock
    except ImportError:
        import sys
        from ditk import logging
        logging.warning("Please install readerwriterlock first, such as `pip3 install readerwriterlock`.")
        sys.exit(1)
    if name not in rw_lock_mapping:
        rw_lock_mapping[name] = rwlock.RWLockFairD()
    lock = rw_lock_mapping[name]
    if op == 'read':
        return lock.gen_rlock()
    elif op == 'write':
        return lock.gen_wlock()


class FcntlContext:
    """
    Overview:
        A context manager that acquires an exclusive lock on a file using fcntl. \
        This is useful for preventing multiple processes from running the same code.

    Interfaces:
        ``__init__``, ``__enter__``, ``__exit__``.

    Example:
        >>> lock_path = "/path/to/lock/file"
        >>> with FcntlContext(lock_path) as lock:
        >>>    # Perform operations while the lock is held

    """

    def __init__(self, lock_path: str) -> None:
        """
        Overview:
            Initialize the LockHelper object.

        Arguments:
            - lock_path (:obj:`str`): The path to the lock file.
        """
        self.lock_path = lock_path
        self.f = None

    def __enter__(self) -> None:
        """
        Overview:
            Acquires the lock and opens the lock file in write mode. \
            If the lock file does not exist, it is created.
        """
        assert self.f is None, self.lock_path
        self.f = open(self.lock_path, 'w')
        fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *args, **kwargs) -> None:
        """
        Overview:
            Closes the file and releases any resources used by the lock_helper object.
        Arguments:
            - args (:obj:`Tuple`): The arguments passed to the ``__exit__`` function.
            - kwargs (:obj:`Dict`): The keyword arguments passed to the ``__exit__`` function.
        """
        self.f.close()
        self.f = None


def get_file_lock(name: str, op: str) -> FcntlContext:
    """
    Overview:
        Acquires a file lock for the specified file. \

    Arguments:
        - name (:obj:`str`): The name of the file.
        - op (:obj:`str`): The operation to perform on the file lock.
    """
    if fcntl is None:
        return get_rw_file_lock(name, op)
    else:
        lock_name = name + '.lock'
        if not os.path.isfile(lock_name):
            try:
                Path(lock_name).touch()
            except Exception as e:
                pass
        return FcntlContext(lock_name)

import multiprocessing
import threading
from enum import Enum, unique
from readerwriterlock import rwlock


@unique
class LockContextType(Enum):
    """
    Enum to express the type of the lock
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
        __init__, __enter__, __exit__

    Example:
        >>> with LockContext() as lock:
        >>>     print("Do something here.")
    """

    def __init__(self, type_: LockContextType = LockContextType.THREAD_LOCK):
        r"""
        Overview:
            init the lock according to given type
        """
        self.lock = _LOCK_TYPE_MAPPING[type_]()

    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def __enter__(self):
        """
        Overview:
            Entering the context and acquire lock
        """
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        """
        Overview:
            Quiting the context and release lock
        """
        self.lock.release()


rw_lock_mapping = {}


def get_rw_lock(name: str, op: str):
    assert op in ['read', 'write']
    if name not in rw_lock_mapping:
        rw_lock_mapping[name] = rwlock.RWLockFairD()
    lock = rw_lock_mapping[name]
    if op == 'read':
        return lock.gen_rlock()
    elif op == 'write':
        return lock.gen_wlock()

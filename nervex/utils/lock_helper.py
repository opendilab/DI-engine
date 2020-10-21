import multiprocessing
import threading
from enum import Enum, unique


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
    Generate a LockContext in order to make sure the thread safety.

    Example:
        >>> with LockContext() as lock:
        >>>     print("Do something here.")
    """

    def __init__(self, type_: LockContextType = LockContextType.THREAD_LOCK):
        self.lock = _LOCK_TYPE_MAPPING[type_]()

    def __enter__(self):
        """
        Entering the context
        """
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        """
        Quiting the context
        """
        self.lock.release()

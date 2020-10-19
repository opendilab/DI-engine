import multiprocessing
import threading
from enum import Enum, unique


@unique
class LockContextType(Enum):
    THREAD_LOCK = 1
    PROCESS_LOCK = 2


_LOCK_TYPE_MAPPING = {
    LockContextType.THREAD_LOCK: threading.Lock,
    LockContextType.PROCESS_LOCK: multiprocessing.Lock,
}


class LockContext(object):

    def __init__(self, type_: LockContextType = LockContextType.THREAD_LOCK):
        self.lock = _LOCK_TYPE_MAPPING[type_]()

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        self.lock.release()

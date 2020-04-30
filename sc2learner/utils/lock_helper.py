import threading
import multiprocessing


class LockContext(object):
    def __init__(self, lock_type):
        assert (lock_type in ['thread', 'process'])
        if lock_type == 'thread':
            self.lock = threading.Lock()
        elif lock_type == 'process':
            self.lock = multiprocessing.Lock()

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        self.lock.release()

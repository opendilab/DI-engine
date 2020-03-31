import threading


class LockContext(object):
    def __init__(self, lock_type):
        assert (lock_type in ['thread'])
        if lock_type == 'thread':
            self.lock = threading.Lock()

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        self.lock.release()

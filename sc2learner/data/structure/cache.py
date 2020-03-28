import time
from queue import Queue
from threading import Thread
from sc2learner.utils import LockContext


class Cache(object):
    def __init__(self, maxlen, timeout, monitor_interval=1.0, debug=False):
        assert (maxlen > 0)
        self.maxlen = maxlen
        self.timeout = timeout
        self.monitor_interval = monitor_interval
        self.debug = debug
        self.receive_queue = Queue(maxlen)
        self.send_queue = Queue(maxlen)
        self.receive_lock = LockContext(lock_type='thread')
        self._timeout_thread = Thread(target=self._timeout_monitor)
        self._timeout_thread_flag = True

    def push_data(self, data):
        with self.receive_lock:
            self.receive_queue.put([data, time.time()])
            if self.receive_queue.full():
                self.dprint('send total receive_queue, current len:{}'.format(self.receive_queue.qsize()))
                while not self.receive_queue.empty():
                    self.send_queue.put(self.receive_queue.get()[0])

    def get_cached_data_iter(self):
        return iter(self.send_queue.get, 'STOP')

    def _timeout_monitor(self):
        while self._timeout_thread_flag:
            time.sleep(self.monitor_interval)
            with self.receive_lock:
                while not self.receive_queue.empty():
                    wait_time = time.time() - self.receive_queue.queue[0][1]
                    if wait_time >= self.timeout:
                        self.dprint(
                            'excess the maximum wait time, eject from the cache.(wait_time/timeout: {}/{}'.format(
                                wait_time, self.timeout
                            )
                        )
                        self.send_queue.put(self.receive_queue.get()[0])
                    else:
                        break

    def run(self):
        self._timeout_thread.start()

    def close(self):
        self._timeout_thread_flag = False
        self.send_queue.put('STOP')

    def dprint(self, s):
        if self.debug:
            print('[CACHE] ' + s)

    @property
    def remain_data_count(self):
        return self.receive_queue.qsize()

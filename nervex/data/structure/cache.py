import time
from queue import Queue
from threading import Thread

from nervex.utils import LockContext, LockContextType


class Cache:
    r"""
    Overview:
        data cache for reducing concurrent pressure, with timeout and full queue eject mechanism
    Interface:
        __init__, push_data, get_cached_data_iter, run, close
    Property:
        remain_data_count
    """

    def __init__(self, maxlen, timeout, monitor_interval=1.0, _debug=False):
        r"""
        Overview:
            initialize the cache object
        Arguments:
            - maxlen (:obj:`int`): the maximum length of the cache queue
            - timeout (:obj:`float`): the maximum second of the data can remain in the cache
            - monitor_interval (:obj:`float`): the interval of the timeout monitor thread checks the time
            - _debug (:obj:`bool`): whether to use debug mode, which enables some debug print info
        """
        assert maxlen > 0
        self.maxlen = maxlen
        self.timeout = timeout
        self.monitor_interval = monitor_interval
        self.debug = _debug
        # two seperate receive and send queue for reducing interaction frequency and interference
        self.receive_queue = Queue(maxlen)
        self.send_queue = Queue(maxlen)
        self.receive_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._timeout_thread = Thread(target=self._timeout_monitor)
        # the bool flag for gracefully shutting down the timeout monitor thread
        self._timeout_thread_flag = True

    def push_data(self, data):
        r"""
        Overview:
            push data into receive queue, if the receive queue is full(after push), then push all the data
            into send queue
        Arguments:
            - data (:obj:`T`): the data need to be added into queue

        .. tip::
            thread-safe
        """
        with self.receive_lock:
            # record and save the time of the data pushed into queue
            self.receive_queue.put([data, time.time()])
            if self.receive_queue.full():
                self.dprint('send total receive_queue, current len:{}'.format(self.receive_queue.qsize()))
                while not self.receive_queue.empty():
                    self.send_queue.put(self.receive_queue.get()[0])  # only send raw data to send queue

    def get_cached_data_iter(self):
        r"""
        Overview:
            get the iterator of the send queue, once a data is pushed into send queue, it can be accessed by
            this iterator, 'STOP' is the end flag of this iterator
        Returns:
            - iterator (:obj:`callable_iterator`) the send queue iterator
        """
        return iter(self.send_queue.get, 'STOP')

    def _timeout_monitor(self):
        r"""
        Overview:
            the workflow of the timeout monitor thread
        """
        while self._timeout_thread_flag:  # loop until the flag is set
            time.sleep(self.monitor_interval)  # with a fixed check interval
            with self.receive_lock:
                # for non-empty receive_queue, check the time from the head to the tail(only access no pop) until find
                # the first not timeout data
                while not self.receive_queue.empty():
                    # check the time of the data remains in the receive_queue, if excesses the timeout then returns True
                    is_timeout = self._warn_if_timeout()
                    if not is_timeout:
                        break

    def _warn_if_timeout(self):
        r"""
        Overview:
            return whether is timeout
        Returns
            - result: (:obj:`bool`) whether is timeout
        """
        wait_time = time.time() - self.receive_queue.queue[0][1]
        if wait_time >= self.timeout:
            self.dprint(
                'excess the maximum wait time, eject from the cache.(wait_time/timeout: {}/{}'.format(
                    wait_time, self.timeout
                )
            )
            self.send_queue.put(self.receive_queue.get()[0])
            return True
        else:
            return False

    def run(self):
        r"""
        Overview:
            launch the cache internal thread, e.g. timeout monitor thread
        """
        self._timeout_thread.start()

    def close(self):
        r"""
        Overview:
            shut down the cache internal thread and send the end flag to send queue iterator
        """
        self._timeout_thread_flag = False
        self.send_queue.put('STOP')

    def dprint(self, s):
        if self.debug:
            print('[CACHE] ' + s)

    @property
    def remain_data_count(self):
        r"""
        Overview:
            return the remain data count in the receive queue
        Returns:
            - count (:obj:`int`) the size of the receive queue
        """
        return self.receive_queue.qsize()

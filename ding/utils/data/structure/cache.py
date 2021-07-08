import time
from queue import Queue
from threading import Thread
from typing import Any

from ding.utils import LockContext, LockContextType


class Cache:
    r"""
    Overview:
        Data cache for reducing concurrent pressure, with timeout and full queue eject mechanism
    Interface:
        __init__, push_data, get_cached_data_iter, run, close
    Property:
        remain_data_count
    """

    def __init__(self, maxlen: int, timeout: float, monitor_interval: float = 1.0, _debug: bool = False) -> None:
        r"""
        Overview:
            Initialize the cache object.
        Arguments:
            - maxlen (:obj:`int`): Maximum length of the cache queue.
            - timeout (:obj:`float`): Maximum second of the data can remain in the cache.
            - monitor_interval (:obj:`float`): Interval of the timeout monitor thread checks the time.
            - _debug (:obj:`bool`): Whether to use debug mode or not, which enables debug print info.
        """
        assert maxlen > 0
        self.maxlen = maxlen
        self.timeout = timeout
        self.monitor_interval = monitor_interval
        self.debug = _debug
        # two separate receive and send queue for reducing interaction frequency and interference
        self.receive_queue = Queue(maxlen)
        self.send_queue = Queue(maxlen)
        self.receive_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._timeout_thread = Thread(target=self._timeout_monitor)
        # the bool flag for gracefully shutting down the timeout monitor thread
        self._timeout_thread_flag = True

    def push_data(self, data: Any) -> None:
        r"""
        Overview:
            Push data into receive queue, if the receive queue is full(after push), then push all the data
            in receive queue into send queue.
        Arguments:
            - data (:obj:`Any`): The data which needs to be added into receive queue

        .. tip::
            thread-safe
        """
        with self.receive_lock:
            # Push the data item and current time together into queue
            self.receive_queue.put([data, time.time()])
            if self.receive_queue.full():
                self.dprint('send total receive_queue, current len:{}'.format(self.receive_queue.qsize()))
                while not self.receive_queue.empty():
                    # Only send raw data to send queue
                    self.send_queue.put(self.receive_queue.get()[0])

    def get_cached_data_iter(self) -> 'callable_iterator':  # noqa
        r"""
        Overview:
            Get the iterator of the send queue. Once a data is pushed into send queue, it can be accessed by
            this iterator. 'STOP' is the end flag of this iterator.
        Returns:
            - iterator (:obj:`callable_iterator`) The send queue iterator.
        """
        return iter(self.send_queue.get, 'STOP')

    def _timeout_monitor(self) -> None:
        r"""
        Overview:
            The workflow of the timeout monitor thread.
        """
        # Loop until the flag is set to False
        while self._timeout_thread_flag:
            # A fixed check interval
            time.sleep(self.monitor_interval)
            with self.receive_lock:
                # For non-empty receive_queue, check the time from head to tail(only access no pop) until finding
                # the first data which is not timeout
                while not self.receive_queue.empty():
                    # Check the time of the data remains in the receive_queue, if excesses the timeout then returns True
                    is_timeout = self._warn_if_timeout()
                    if not is_timeout:
                        break

    def _warn_if_timeout(self) -> bool:
        r"""
        Overview:
            Return whether is timeout.
        Returns
            - result: (:obj:`bool`) Whether is timeout.
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

    def run(self) -> None:
        r"""
        Overview:
            Launch the cache internal thread, e.g. timeout monitor thread.
        """
        self._timeout_thread.start()

    def close(self) -> None:
        r"""
        Overview:
            Shut down the cache internal thread and send the end flag to send queue's iterator.
        """
        self._timeout_thread_flag = False
        self.send_queue.put('STOP')

    def dprint(self, s: str) -> None:
        r"""
        Overview:
            In debug mode, print debug str.
        Arguments:
            - s (:obj:`str`): Debug info to be printed.
        """
        if self.debug:
            print('[CACHE] ' + s)

    @property
    def remain_data_count(self) -> int:
        r"""
        Overview:
            Return receive queue's remain data count
        Returns:
            - count (:obj:`int`): The size of the receive queue.
        """
        return self.receive_queue.qsize()

from threading import Event, Lock
from typing import Optional


class DblEvent:

    def __init__(self, opened: bool = False):
        self.__open_event = Event()
        self.__close_event = Event()
        self.__lock = Lock()

        if opened:
            self.__open_event.set()
        else:
            self.__close_event.set()

    def wait_for_open(self, timeout: Optional[float] = None):
        self.__open_event.wait(timeout=timeout)

    def wait_for_close(self, timeout: Optional[float] = None):
        self.__close_event.wait(timeout=timeout)

    def open(self):
        with self.__lock:
            self.__open_event.set()
            self.__close_event.clear()

    def close(self):
        with self.__lock:
            self.__close_event.set()
            self.__open_event.clear()

    def is_open(self) -> bool:
        with self.__lock:
            return self.__open_event.is_set()

    def is_close(self) -> bool:
        with self.__lock:
            return self.__close_event.is_set()

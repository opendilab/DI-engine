from threading import Event, Lock
from typing import Optional


class DblEvent:
    """
    Overview:
        A double event object, can open and close.
        Bases on 2 event objects
    """

    def __init__(self, opened: bool = False):
        """
        Overview:
            Constructor of `DblEvent`
        Arguments:
            - opened (:obj:`bool`): Initial status (`True` means open, `False` means close, default is `False`)
        """
        self.__open_event = Event()
        self.__close_event = Event()
        self.__lock = Lock()

        if opened:
            self.__open_event.set()
        else:
            self.__close_event.set()

    def wait_for_open(self, timeout: Optional[float] = None):
        """
        Overview:
            Wait until the event is opened
        Arguments:
            - timeout (:obj:`Optional[float]`): Waiting time out in seconds
        """
        self.__open_event.wait(timeout=timeout)

    def wait_for_close(self, timeout: Optional[float] = None):
        """
        Overview:
            Wait until the event is closed
        Arguments:
            - timeout (:obj:`Optional[float]`): Waiting time out in seconds
        """
        self.__close_event.wait(timeout=timeout)

    def open(self):
        """
        Overview:
            Open this event
        """
        with self.__lock:
            self.__open_event.set()
            self.__close_event.clear()

    def close(self):
        """
        Overview:
            Close this event
        """
        with self.__lock:
            self.__close_event.set()
            self.__open_event.clear()

    def is_open(self) -> bool:
        """
        Overview:
            Get if the event is opened
        Returns:
            - opened (:obj:`bool`): The event is opened or not
        """
        with self.__lock:
            return self.__open_event.is_set()

    def is_close(self) -> bool:
        """
        Overview:
            Get if the event is closed
        Returns:
            - opened (:obj:`bool`): The event is closed or not
        """
        with self.__lock:
            return self.__close_event.is_set()

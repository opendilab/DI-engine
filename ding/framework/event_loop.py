from collections import defaultdict
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import fnmatch
from ditk import logging


class EventLoop:
    loops = {}

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._listeners = defaultdict(list)
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        self._exception = None
        self._active = True

    def on(self, event: str, fn: Callable) -> None:
        """
        Overview:
            Subscribe to an event, execute this function every time the event is emitted.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): The function.
        """
        self._listeners[event].append(fn)

    def off(self, event: str, fn: Optional[Callable] = None) -> None:
        """
        Overview:
            Unsubscribe an event, or a specific function in the event.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Optional[Callable]`): The function.
        """
        for e in fnmatch.filter(self._listeners.keys(), event):
            if fn:
                self._listeners[e].remove(fn)
            else:
                self._listeners[e] = []

    def once(self, event: str, fn: Callable) -> None:
        """
        Overview:
            Subscribe to an event, execute this function only once when the event is emitted.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): The function.
        """

        def once_callback(*args, **kwargs):
            self.off(event, once_callback)
            fn(*args, **kwargs)

        self.on(event, once_callback)

    def emit(self, event: str, *args, **kwargs) -> None:
        """
        Overview:
            Emit an event, call listeners.
            If there is an unhandled error in this event loop, calling emit will raise an exception,
            which will cause the process to exit.
        Arguments:
            - event (:obj:`str`): Event name.
        """
        if self._exception:
            raise self._exception
        if self._active:
            self._thread_pool.submit(self._trigger, event, *args, **kwargs)

    def _trigger(self, event: str, *args, **kwargs) -> None:
        """
        Overview:
            Execute the callbacks under the event. If any callback raise an exception,
            we will save the traceback and ignore the exception.
        Arguments:
            - event (:obj:`str`): Event name.
        """
        if event not in self._listeners:
            logging.debug("Event {} is not registered in the callbacks of {}!".format(event, self._name))
            return
        for fn in self._listeners[event]:
            try:
                fn(*args, **kwargs)
            except Exception as e:
                self._exception = e

    def listened(self, event: str) -> bool:
        """
        Overview:
            Check if the event has been listened to.
        Arguments:
            - event (:obj:`str`): Event name
        Returns:
            - listened (:obj:`bool`): Whether this event has been listened to.
        """
        return event in self._listeners

    @classmethod
    def get_event_loop(cls: type, name: str = "default") -> "EventLoop":
        """
        Overview:
            Get new event loop when name not exists, or return the existed instance.
        Arguments:
            - name (:obj:`str`): Name of event loop.
        """
        if name in cls.loops:
            return cls.loops[name]
        cls.loops[name] = loop = cls(name)
        return loop

    def stop(self) -> None:
        self._active = False
        self._listeners = defaultdict(list)
        self._exception = None
        self._thread_pool.shutdown()
        if self._name in EventLoop.loops:
            del EventLoop.loops[self._name]

    def __del__(self) -> None:
        if self._active:
            self.stop()

from collections import defaultdict
import logging
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor


class EventLoop:
    loops = {}

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._listeners = defaultdict(list)
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        self._exc = None
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
        if fn:
            self._listeners[event].remove(fn)
        else:
            self._listeners[event] = []

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
        if self._exc:
            raise self._exc
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
            logging.warning("Event {} is not registered in the callbacks!".format(event))
            return
        for fn in self._listeners[event]:
            try:
                fn(*args, **kwargs)
            except Exception as e:
                self._exc = e

    @staticmethod
    def get_event_loop(name: str = "default") -> "EventLoop":
        """
        Overview:
            Get new event loop when name not exists, or return the existed instance.
        Arguments:
            - name (:obj:`str`): Name of event loop.
        """
        if name in EventLoop.loops:
            return EventLoop.loops[name]
        EventLoop.loops[name] = loop = EventLoop(name)
        return loop

    def stop(self):
        self._active = False
        self._listeners = defaultdict(list)
        self._exc = None
        self._thread_pool.shutdown()
        if self._name in EventLoop.loops:
            del EventLoop.loops[self._name]

    @staticmethod
    def stop_event_loop(name: str) -> None:
        """
        Overview:
            Stop and delete event loop from global instances list.
        Arguments:
            - name (:obj:`str`): Name of event loop.
        """
        if name in EventLoop.loops:
            EventLoop.loops[name].stop()

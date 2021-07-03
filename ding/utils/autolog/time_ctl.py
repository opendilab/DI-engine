import time
from abc import ABCMeta, abstractmethod
from typing import Union

from ..lock_helper import LockContext, LockContextType


class BaseTime(metaclass=ABCMeta):
    """
    Overview:
        Abstract time interface
    """

    @abstractmethod
    def time(self) -> Union[int, float]:
        """
        Overview:
            Get time information

        Returns:
            - time(:obj:`float, int`): time information
        """
        raise NotImplementedError


class NaturalTime(BaseTime):
    """
    Overview:
        Natural time object

    Example:
        >>> from ding.utils.autolog.time_ctl import NaturalTime
        >>> time_ = NaturalTime()
    """

    def __init__(self):
        self.__last_time = None

    def time(self) -> float:
        """
        Overview:
            Get current natural time (float format, unix timestamp)

        Returns:
            - time(:obj:`float`): unix timestamp

        Example:
            >>> from ding.utils.autolog.time_ctl import NaturalTime
            >>> time_ = NaturalTime()
            >>> time_.time()
            1603896383.8811457
        """
        _current_time = time.time()
        if self.__last_time is not None:
            _current_time = max(_current_time, self.__last_time)

        self.__last_time = _current_time
        return _current_time


class TickTime(BaseTime):
    """
    Overview:
        Tick time object

    Example:
        >>> from ding.utils.autolog.time_ctl import TickTime
        >>> time_ = TickTime()
    """

    def __init__(self, init: int = 0):
        """
        Overview:
            Constructor of TickTime

        Args:
            init (int, optional): init tick time, default is 1
        """
        self.__tick_time = init

    def step(self, delta: int = 1) -> int:
        """
        Overview
            Step the time forward for this TickTime

        Args:
             delta (int, optional): steps to step forward, default is 1

        Returns:
            int: new time after stepping

        Example:
            >>> from ding.utils.autolog.time_ctl import TickTime
            >>> time_ = TickTime(0)
            >>> time_.step()
            1
            >>> time_.step(2)
            3
        """
        if not isinstance(delta, int):
            raise TypeError("Delta should be positive int, but {actual} found.".format(actual=type(delta).__name__))
        elif delta < 1:
            raise ValueError("Delta should be no less than 1, but {actual} found.".format(actual=repr(delta)))
        else:
            self.__tick_time += delta
            return self.__tick_time

    def time(self) -> int:
        """
        Overview
            Get current tick time

        Returns:
            int: current tick time

        Example:
            >>> from ding.utils.autolog.time_ctl import TickTime
            >>> time_ = TickTime(0)
            >>> time_.step()
            >>> time_.time()
            1
        """
        return self.__tick_time


class TimeProxy(BaseTime):
    """
    Overview:
        Proxy of time object, it can freeze time, sometimes useful when reproducing.
        This object is thread-safe, and also freeze and unfreeze operation is strictly ordered.

    Example:
        >>> from ding.utils.autolog.time_ctl import TickTime, TimeProxy
        >>> tick_time_ = TickTime()
        >>> time_ = TimeProxy(tick_time_)
        >>> tick_time_.step()
        >>> print(tick_time_.time(), time_.time(), time_.current_time())
        1 1 1
        >>> time_.freeze()
        >>> tick_time_.step()
        >>> print(tick_time_.time(), time_.time(), time_.current_time())
        2 1 2
        >>> time_.unfreeze()
        >>> print(tick_time_.time(), time_.time(), time_.current_time())
        2 2 2
    """

    def __init__(self, time_: BaseTime, frozen: bool = False, lock_type: LockContextType = LockContextType.THREAD_LOCK):
        """
        Overview:
            Constructor for Time proxy

        Args:
            time_ (BaseTime): another time object it based on
            frozen (bool, optional): this object will be frozen immediately if true, otherwise not, default is False
            lock_type (LockContextType, optional): type of the lock, default is THREAD_LOCK
        """
        self.__time = time_
        self.__current_time = self.__time.time()

        self.__frozen = frozen
        self.__lock = LockContext(lock_type)
        self.__frozen_lock = LockContext(lock_type)
        if self.__frozen:
            self.__frozen_lock.acquire()

    @property
    def is_frozen(self) -> bool:
        """
        Overview:
            Get if this time proxy object is frozen

        Returns:
            bool: true if it is frozen, otherwise false
        """
        with self.__lock:
            return self.__frozen

    def freeze(self):
        """
        Overview:
            Freeze this time proxy
        """
        with self.__lock:
            self.__frozen_lock.acquire()
            self.__frozen = True
            self.__current_time = self.__time.time()

    def unfreeze(self):
        """
        Overview:
            Unfreeze this time proxy
        """
        with self.__lock:
            self.__frozen = False
            self.__frozen_lock.release()

    def time(self) -> Union[int, float]:
        """
        Overview:
            Get time (may be frozen time)

        Returns:
            int or float: the time
        """
        with self.__lock:
            if self.__frozen:
                return self.__current_time
            else:
                return self.__time.time()

    def current_time(self) -> Union[int, float]:
        """
        Overview:
            Get current time (will not be frozen time)

        Returns:
            int or float: current time
        """
        return self.__time.time()

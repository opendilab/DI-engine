import pickle
from abc import abstractmethod, ABCMeta
from collections import deque
from threading import Lock
from typing import TypeVar, Iterable

from .time_ctl import BaseTime
from ..collection_helper import iter_mapping

_Tp = TypeVar('_Tp')


class RangedData(metaclass=ABCMeta):
    def __init__(self, expire: float):
        self.__expire = expire
        self.__check_expire()

        self.__data_max_id = 0
        self.__data_items = {}
        self.__data_lock = Lock()

        self.__queue = deque()
        self.__lock = Lock()

    def __check_expire(self):
        if isinstance(self.__expire, (int, float)):
            if self.__expire <= 0:
                raise ValueError(
                    "Expire should be greater than 0, but {actual} found.".format(actual=repr(self.__expire)))
        else:
            raise TypeError(
                'Expire should be int or float, but {actual} found.'.format(actual=type(self.__expire).__name__))

    def __registry_data_item(self, data: _Tp) -> int:
        with self.__data_lock:
            self.__data_max_id += 1
            self.__data_items[self.__data_max_id] = pickle.dumps(data)

            return self.__data_max_id

    def __get_data_item(self, data_id: int) -> _Tp:
        with self.__data_lock:
            return pickle.loads(self.__data_items[data_id])

    def __remove_data_item(self, data_id: int):
        with self.__data_lock:
            del self.__data_items[data_id]

    def __check_time(self, time_: float):
        if self.__queue:
            _time, _ = self.__queue[-1]
            if time_ < _time:
                raise ValueError("Time {time} invalid for descending from last time {last_time}".format(
                    time=repr(time_), last_time=repr(_time)
                ))

    def __append_item(self, time_: float, data: _Tp):
        self.__queue.append((time_, self.__registry_data_item(data)))

    def __flush_history(self):
        _time = self._get_time()
        _limit_time = _time - self.__expire
        while self.__queue:
            _head_time, _head_id = self.__queue.popleft()
            if _head_time >= _limit_time:
                self.__queue.appendleft((_head_time, _head_id))
                break
            else:
                self.__remove_data_item(_head_id)

    def __append(self, time_: float, data: _Tp):
        self.__check_time(time_)
        self.__append_item(time_, data)
        self.__flush_history()

    def __current(self):
        if self.__queue:
            _tail_time, _tail_id = self.__queue.pop()
            self.__queue.append((_tail_time, _tail_id))
            return self.__get_data_item(_tail_id)
        else:
            raise ValueError("This range is empty.")

    def append(self, data: _Tp):
        with self.__lock:
            self.__flush_history()
            _time = self._get_time()
            self.__append(_time, data)
            return self

    def extend(self, iter_: Iterable[_Tp]):
        with self.__lock:
            self.__flush_history()
            _time = self._get_time()
            for item in iter_:
                self.__append(_time, item)
            return self

    def current(self):
        with self.__lock:
            self.__flush_history()
            return self.__current()

    @property
    def expire(self) -> float:
        with self.__lock:
            self.__flush_history()
            return self.__expire

    def __len__(self):
        with self.__lock:
            self.__flush_history()
            return len(self.__data_items)

    def __bool__(self):
        with self.__lock:
            self.__flush_history()
            return not not self.__queue

    def __iter__(self):
        with self.__lock:
            self.__flush_history()
            return iter_mapping(self.__queue, lambda t: self.__get_data_item(t[1]))

    def __getitem__(self, item):
        with self.__lock:
            self.__flush_history()
            _, _id = self.__queue.__getitem__(item)
            return self.__get_data_item(_id)

    @abstractmethod
    def _get_time(self) -> float:
        raise NotImplementedError


class TimeRangedData(RangedData):
    def __init__(self, time_: BaseTime, expire: float):
        RangedData.__init__(self, expire)
        self.__time = time_

    def _get_time(self) -> float:
        return self.__time.time()

    @property
    def time(self):
        return self.__time

import pickle
from abc import abstractmethod, ABCMeta
from collections import deque
from threading import Lock
from typing import TypeVar, Iterable, List, Tuple, Union

from .time_ctl import BaseTime

_Tp = TypeVar('_Tp')


class RangedData(metaclass=ABCMeta):
    """
    Overview:
        A data structure that can store data for a period of time.
    Interfaces:
        ``__init__``, ``append``, ``extend``, ``current``, ``history``, ``expire``, ``__bool__``, ``_get_time``.
    Properties:
        - expire (:obj:`float`): The expire time.
    """

    def __init__(self, expire: float, use_pickle: bool = False):
        """
        Overview:
            Initialize the RangedData object.
        Arguments:
            - expire (:obj:`float`): The expire time of the data.
            - use_pickle (:obj:`bool`): Whether to use pickle to serialize the data.
        """

        self.__expire = expire
        self.__use_pickle = use_pickle
        self.__check_expire()

        self.__data_max_id = 0
        self.__data_items = {}
        self.__data_lock = Lock()

        self.__last_item = None
        self.__queue = deque()
        self.__lock = Lock()

    def __check_expire(self):
        """
        Overview:
            Check the expire time.
        """

        if isinstance(self.__expire, (int, float)):
            if self.__expire <= 0:
                raise ValueError(
                    "Expire should be greater than 0, but {actual} found.".format(actual=repr(self.__expire))
                )
        else:
            raise TypeError(
                'Expire should be int or float, but {actual} found.'.format(actual=type(self.__expire).__name__)
            )

    def __registry_data_item(self, data: _Tp) -> int:
        """
        Overview:
            Registry the data item.
        Arguments:
            - data (:obj:`_Tp`): The data item.
        """

        with self.__data_lock:
            self.__data_max_id += 1
            if self.__use_pickle:
                self.__data_items[self.__data_max_id] = pickle.dumps(data)
            else:
                self.__data_items[self.__data_max_id] = data

            return self.__data_max_id

    def __get_data_item(self, data_id: int) -> _Tp:
        """
        Overview:
            Get the data item.
        Arguments:
            - data_id (:obj:`int`): The data id.
        """

        with self.__data_lock:
            if self.__use_pickle:
                return pickle.loads(self.__data_items[data_id])
            else:
                return self.__data_items[data_id]

    def __remove_data_item(self, data_id: int):
        """
        Overview:
            Remove the data item.
        Arguments:
            - data_id (:obj:`int`): The data id.
        """

        with self.__data_lock:
            del self.__data_items[data_id]

    def __check_time(self, time_: float):
        """
        Overview:
            Check the time.
        Arguments:
            - time_ (:obj:`float`): The time.
        """

        if self.__queue:
            _time, _ = self.__queue[-1]
            if time_ < _time:
                raise ValueError(
                    "Time {time} invalid for descending from last time {last_time}".format(
                        time=repr(time_), last_time=repr(_time)
                    )
                )

    def __append_item(self, time_: float, data: _Tp):
        """
        Overview:
            Append the data item.
        Arguments:
            - time_ (:obj:`float`): The time.
            - data (:obj:`_Tp`): The data item.
        """

        self.__queue.append((time_, self.__registry_data_item(data)))

    def __flush_history(self):
        """
        Overview:
            Flush the history data.
        """

        _time = self._get_time()
        _limit_time = _time - self.__expire
        while self.__queue:
            _head_time, _head_id = self.__queue.popleft()
            if _head_time >= _limit_time:
                self.__queue.appendleft((_head_time, _head_id))
                break
            else:
                if self.__last_item:
                    _last_time, _last_id = self.__last_item
                    self.__remove_data_item(_last_id)

                self.__last_item = (_head_time, _head_id)

    def __append(self, time_: float, data: _Tp):
        """
        Overview:
            Append the data.
        """

        self.__check_time(time_)
        self.__append_item(time_, data)
        self.__flush_history()

    def __current(self):
        """
        Overview:
            Get the current data.
        """

        if self.__queue:
            _tail_time, _tail_id = self.__queue.pop()
            self.__queue.append((_tail_time, _tail_id))
            return self.__get_data_item(_tail_id)
        elif self.__last_item:
            _last_time, _last_id = self.__last_item
            return self.__get_data_item(_last_id)
        else:
            raise ValueError("This range is empty.")

    def __history_yield(self):
        """
        Overview:
            Yield the history data.
        """

        _time = self._get_time()
        _limit_time = _time - self.__expire
        _latest_time, _latest_id = None, None

        if self.__last_item:
            _latest_time, _latest_id = _last_time, _last_id = self.__last_item
            yield max(_last_time, _limit_time), self.__get_data_item(_last_id)

        for _item_time, _item_id in self.__queue:
            _latest_time, _latest_id = _item_time, _item_id
            yield _item_time, self.__get_data_item(_item_id)

        if _latest_time is not None and _latest_time < _time:
            yield _time, self.__get_data_item(_latest_id)

    def __history(self):
        """
        Overview:
            Get the history data.
        """

        return list(self.__history_yield())

    def append(self, data: _Tp):
        """
        Overview:
            Append the data.
        """

        with self.__lock:
            self.__flush_history()
            _time = self._get_time()
            self.__append(_time, data)
            return self

    def extend(self, iter_: Iterable[_Tp]):
        """
        Overview:
            Extend the data.
        """

        with self.__lock:
            self.__flush_history()
            _time = self._get_time()
            for item in iter_:
                self.__append(_time, item)
            return self

    def current(self) -> _Tp:
        """
        Overview:
            Get the current data.
        """

        with self.__lock:
            self.__flush_history()
            return self.__current()

    def history(self) -> List[Tuple[Union[int, float], _Tp]]:
        """
        Overview:
            Get the history data.
        """

        with self.__lock:
            self.__flush_history()
            return self.__history()

    @property
    def expire(self) -> float:
        """
        Overview:
            Get the expire time.
        """

        with self.__lock:
            self.__flush_history()
            return self.__expire

    def __bool__(self):
        """
        Overview:
            Check whether the range is empty.
        """

        with self.__lock:
            self.__flush_history()
            return not not (self.__queue or self.__last_item)

    @abstractmethod
    def _get_time(self) -> float:
        """
        Overview:
            Get the current time.
        """

        raise NotImplementedError


class TimeRangedData(RangedData):
    """
    Overview:
        A data structure that can store data for a period of time.
    Interfaces:
        ``__init__``, ``_get_time``, ``append``, ``extend``, ``current``, ``history``, ``expire``, ``__bool__``.
    Properties:
        - time (:obj:`BaseTime`): The time.
        - expire (:obj:`float`): The expire time.
    """

    def __init__(self, time_: BaseTime, expire: float):
        """
        Overview:
            Initialize the TimeRangedData object.
        Arguments:
            - time_ (:obj:`BaseTime`): The time.
            - expire (:obj:`float`): The expire time.
        """

        RangedData.__init__(self, expire)
        self.__time = time_

    def _get_time(self) -> float:
        """
        Overview:
            Get the current time.
        """

        return self.__time.time()

    @property
    def time(self):
        """
        Overview:
            Get the time.
        """

        return self.__time

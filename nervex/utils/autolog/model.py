from enum import Enum
from typing import TypeVar, Union, Type, List, Tuple, Any

from .time_ctl import BaseTime, TimeProxy

_TimeType = TypeVar('_TimeType', bound=Union[float, int])
_ValueType = TypeVar('_ValueType')


def _expire_value_records(value_records: List[Tuple[_TimeType, _ValueType]], start_time: _TimeType) \
        -> Tuple[Tuple[_TimeType, _ValueType], List[Tuple[_TimeType, _ValueType]]]:
    max_id = -1
    for i, (_time, _item) in enumerate(value_records):
        if _time >= start_time:
            break
        max_id = i

    if max_id >= 0:
        last_dropped_record = value_records[max_id]
        new_value_records = value_records[max_id + 1:]
    else:
        last_dropped_record = None
        new_value_records = value_records

    return last_dropped_record, new_value_records


class TimeMode(Enum):
    """
    Overview:
        Mode that used to decide the format of range_values function

        ABSOLUTE: use absolute time
        RELATIVE_LIFECYCLE: use relative time based on property's lifecycle
        RELATIVE_CURRENT_TIME: use relative time based on current time
    """
    ABSOLUTE = 0
    RELATIVE_LIFECYCLE = 1
    RELATIVE_CURRENT_TIME = 2


class LoggedValue:
    """
    Overview:
        LoggedValue can be used as property in LoggedModel, for it has __get__ and __set__ method.
        This class's instances will be associated with their owner LoggedModel instance, all the LoggedValue
        of one LoggedModel will shared the only one time object (defined in time_ctl), so that timeline can
        be managed properly.
    """

    def __init__(self, name: str, type_: Type[_ValueType] = object):
        """
        Overview:
            Constructor of LoggedValue
            ATTENTION, default value is not supported and WILL NEVER be supported in constructor,
            for LoggedValue need to bind owner LoggerModel's time object, and this binding operation
            is processed when first time __get__ method is called.

        Args:
            name (str): name of this property
            type_ (type, optional): type limit of this property, default is object (can be seen as no limit)
        """
        self.__instance = None
        self.__life_start_time = None

        self.__name = name
        self.__type = type_
        self.__value = None

        self.__value_history = []
        self.__last_dropped_record = None

    # getter and setter
    def __get__(self, instance: 'LoggedModel', owner: Type['LoggedModel']) -> _ValueType:
        if self.__instance is not None:
            return self.__value
        else:
            raise ValueError("Value not initialized, you should assign it an initial value.")

    def __set__(self, instance: 'LoggedModel', value: _ValueType):
        self.__register_instance(instance)
        _original_value, self.__value = self.__value, value
        try:
            self.__check_value_type()
        except Exception as err:
            self.__value = _original_value
            raise err
        else:
            self.__append_value(self.__value)

    # hook functions for instance
    def __get_range_values(self, mode: TimeMode = TimeMode.RELATIVE_LIFECYCLE) \
            -> List[Tuple[Tuple[_TimeType, _TimeType], _ValueType]]:
        self.__flush_history()
        _current_time, _start_time = self.__get_time()

        _result = []

        def _append(begin: _TimeType, end: _TimeType, value: _ValueType):
            _result.append(((begin, end), value))

        if self.__last_dropped_record and self.__value_history:
            _, _last_dropped_value = self.__last_dropped_record
            _first_time, _ = self.__value_history[0]
            _append(_start_time, _first_time, _last_dropped_value)

        _length = len(self.__value_history)
        for i, (_time, _value) in zip(range(_length), self.__value_history):
            if i + 1 < _length:
                _next_time, _ = self.__value_history[i + 1]
                _append(_time, _next_time, _value)
            else:
                _append(_time, _current_time, _value)

        if mode == TimeMode.RELATIVE_LIFECYCLE:
            _rel_time = self.__life_start_time
        elif mode == TimeMode.RELATIVE_CURRENT_TIME:
            (_, _rel_time), _ = _result[-1]
        else:
            _rel_time = None

        if _rel_time is not None:
            _result = [((_b - _rel_time, _e - _rel_time), _v) for (_b, _e), _v in _result]

        return _result

    # self used private functions
    def __register_instance(self, instance: 'LoggedModel'):
        if self.__instance is None:
            self.__instance = instance
            self.__life_start_time = self.__instance.current_time()

            self.__instance.register_attribute_value('range_values', self.__name, self.__get_range_values)

    def __check_value_type(self):
        if not isinstance(self.__value, self.__type):
            raise TypeError(
                "Type not match, {expect} expect but {actual} found.".format(
                    expect=self.__type.__name__,
                    actual=type(self.__value).__name__,
                )
            )

    def __get_time(self) -> Tuple[_TimeType, _TimeType]:
        _current_time = self.__instance.current_time()
        _start_time = _current_time - self.__instance.expire

        return _current_time, _start_time

    def __append_value(self, value: _ValueType):
        self.__value_history.append((self.__instance.current_time(), value))
        self.__flush_history()

    def __flush_history(self):
        _current_time, _start_time = self.__get_time()
        _last_dropped_record, _new_value_history = \
            _expire_value_records(self.__value_history, start_time=_start_time)

        if _last_dropped_record:
            self.__last_dropped_record = _last_dropped_record
            self.__value_history = _new_value_history


_TimeObjectType = TypeVar('_TimeObjectType', bound=BaseTime)


class LoggedModel:
    """
    Overview:
        A model with timeline (integered time, such as 1st, 2nd, 3rd, can also be modeled as a kind
        of self-defined discrete time, such as the implement of TickTime). Serveral values have association
        with each other can be maintained together by using LoggedModel.

    Example:
        Define AvgList model like this

        >>> from nervex.utils.autolog import LoggedValue, LoggedModel
        >>> class AvgList(LoggedModel):
        >>>     value = LoggedValue('value', float)
        >>>     __property_names = ['value']
        >>>
        >>>     def __init__(self, time_: BaseTime, expire: Union[int, float]):
        >>>         LoggedModel.__init__(self, time_, expire)
        >>>         # attention, original value must be set in __init__ function, or it will not
        >>>         #be activated, the timeline of this value will also be unexpectedly effected.
        >>>         self.value = 0.0
        >>>
        >>>         self.__register()
        >>>
        >>>     def __register(self):
        >>>         def __avg_func(prop_name: str) -> float:  # function to calculate average value of properties
        >>>             records = self.range_values[prop_name]()
        >>>             (_start_time, _), _ = records[0]
        >>>             (_, _end_time), _ = records[-1]
        >>>
        >>>             _duration = _end_time - _start_time
        >>>             _sum = sum([_value * (_end_time - _begin_time) for (_begin_time, _end_time), _value in records])
        >>>
        >>>             return _sum / _duration
        >>>
        >>>         for _prop_name in self.__property_names:
        >>>             self.register_attribute_value('avg', _prop_name, lambda: __avg_func(_prop_name))

        Use it like this

        >>> from nervex.utils.autolog import NaturalTime, TimeMode
        >>>
        >>> if __name__ == "__main__":
        >>>     _time = NaturalTime()
        >>>     ll = AvgList(_time, expire=10)
        >>>
        >>>     # just do something here ...
        >>>
        >>>     print(ll.range_values['value']()) # original range_values function in LoggedModel of last 10 secs
        >>>     print(ll.range_values['value'](TimeMode.ABSOLUTE))  # use absolute time
        >>>     print(ll.avg['value']())  # average value of last 10 secs
    """

    def __init__(self, time_: _TimeObjectType, expire: _TimeType):
        self.__time = time_
        self.__time_proxy = TimeProxy(
            self.__time,
            frozen=False,
        )
        self.__expire = expire

        self.__methods = {}

    @property
    def time(self) -> _TimeObjectType:
        """
        Overview:
            Get original time object passed in, can execute method (such as step()) by this property.

        Returns:
            BaseTime: time object used by this model
        """
        return self.__time

    @property
    def expire(self) -> _TimeType:
        """
        Overview:
            Get expire time

        Returns:
            int or float: time that old value records expired
        """
        return self.__expire

    def fixed_time(self) -> Union[float, int]:
        """
        Overview:
            Get fixed time (will be frozen time if time proxy is frozen)
            This feature can be useful when adding value replay feature (in the future)

        Returns:
            int or float: fixed time
        """
        return self.__time_proxy.time()

    def current_time(self) -> Union[float, int]:
        """
        Overview:
            Get current time (real time that regardless of time proxy's frozen statement)

        Returns:
            int or float: current time
        """
        return self.__time_proxy.current_time()

    def freeze(self):
        """
        Overview:
            Freeze time proxy object.
            This feature can be useful when adding value replay feature (in the future)
        """
        self.__time_proxy.freeze()

    def unfreeze(self):
        """
        Overview:
            Unfreeze time proxy object.
            This feature can be useful when adding value replay feature (in the future)
        """
        self.__time_proxy.unfreeze()

    def register_attribute_value(self, attribute_name: str, property_name: str, value: Any):
        """
        Overview:
            Register a new attribute for one of the values. Example can be found in overview of class.
        """
        self.__methods[attribute_name] = self.__methods.get(attribute_name, {})
        self.__methods[attribute_name][property_name] = value

    def __getattr__(self, attribute_name: str) -> Any:
        """
        Overview:
            Support all methods registered.

        Args:
            attribute_name (str): name of attribute

        Return:
            A indelible object that can return attribute value.

        Example:
            >>> ll = AvgList(NaturalTime(), expire=10)
            >>> ll.range_value['value']  # get 'range_value' attribute of 'value' property, it should be a function
        """
        if attribute_name in self.__methods.keys():
            _attributes = self.__methods[attribute_name]

            class _Cls:

                def __getitem__(self, property_name: str):
                    if property_name in _attributes.keys():
                        return _attributes[property_name]
                    else:
                        raise KeyError(
                            "Attribute {attr_name} for property {prop_name} not found.".format(
                                attr_name=repr(attribute_name),
                                prop_name=repr(property_name),
                            )
                        )

            return _Cls()
        else:
            raise KeyError("Attribute {name} not found.".format(name=repr(attribute_name)))

from typing import TypeVar, Union, List, Any

from .base import _LOGGED_MODEL__PROPERTIES, \
    _LOGGED_MODEL__PROPERTY_ATTR_PREFIX, _TimeType, TimeMode
from .data import TimeRangedData
from .time_ctl import BaseTime, TimeProxy

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
        self.__time_proxy = TimeProxy(self.__time, frozen=False)
        self.__init_time = self.__time_proxy.time()
        self.__expire = expire

        self.__methods = {}

        for name in self.__properties:
            setattr(self, _LOGGED_MODEL__PROPERTY_ATTR_PREFIX + name,
                    TimeRangedData(self.__time_proxy, expire=self.__expire))

    @property
    def __properties(self) -> List[str]:
        return getattr(self, _LOGGED_MODEL__PROPERTIES)

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

    def __get_property_ranged_data(self, name: str) -> TimeRangedData:
        return getattr(self, _LOGGED_MODEL__PROPERTY_ATTR_PREFIX + name)

    def __get_range_values_func(self, name: str):
        def _func(mode: TimeMode):
            _current_time = self.__time_proxy.time()
            _result = self.__get_property_ranged_data(name).history()

            if mode == TimeMode.RELATIVE_LIFECYCLE:
                _result = [(_time - self.__init_time, _data) for _time, _data in _result]
            elif mode == TimeMode.RELATIVE_CURRENT_TIME:
                _result = [(_time - _current_time, _data) for _time, _data in _result]

            return _result

        return _func

    def __register_default_funcs(self):
        for name in self.__properties:
            self.register_attribute_value('range_values', name, self.__get_range_values_func(name))

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

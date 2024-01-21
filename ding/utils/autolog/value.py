from typing import Type

from .base import _LOGGED_VALUE__PROPERTY_NAME, _LOGGED_MODEL__PROPERTY_ATTR_PREFIX, _ValueType
from .data import TimeRangedData


class LoggedValue:
    """
    Overview:
        LoggedValue can be used as property in LoggedModel, for it has __get__ and __set__ method.
        This class's instances will be associated with their owner LoggedModel instance, all the LoggedValue
        of one LoggedModel will shared the only one time object (defined in time_ctl), so that timeline can
        be managed properly.
    Interfaces:
        ``__init__``, ``__get__``, ``__set__``
    Properties:
        - __property_name (:obj:`str`): The name of the property.
    """

    def __init__(self, type_: Type[_ValueType] = object):
        """
        Overview:
            Initialize the LoggedValue object.
        Interfaces:
            ``__init__``
        """

        self.__type = type_

    @property
    def __property_name(self):
        """
        Overview:
            Get the name of the property.
        """

        return getattr(self, _LOGGED_VALUE__PROPERTY_NAME)

    def __get_ranged_data(self, instance) -> TimeRangedData:
        """
        Overview:
            Get the ranged data.
        Interfaces:
            ``__get_ranged_data``
        """

        return getattr(instance, _LOGGED_MODEL__PROPERTY_ATTR_PREFIX + self.__property_name)

    def __get__(self, instance, owner):
        """
        Overview:
            Get the value.
        Arguments:
            - instance (:obj:`LoggedModel`): The owner LoggedModel instance.
            - owner (:obj:`type`): The owner class.
        """

        return self.__get_ranged_data(instance).current()

    def __set__(self, instance, value: _ValueType):
        """
        Overview:
            Set the value.
        Arguments:
            - instance (:obj:`LoggedModel`): The owner LoggedModel instance.
            - value (:obj:`_ValueType`): The value to set.
        """

        if isinstance(value, self.__type):
            return self.__get_ranged_data(instance).append(value)
        else:
            raise TypeError(
                'New value should be {expect}, but {actual} found.'.format(
                    expect=self.__type.__name__,
                    actual=type(value).__name__,
                )
            )

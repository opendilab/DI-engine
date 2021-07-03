from enum import unique, IntEnum
from typing import TypeVar, Union

_LOGGED_VALUE__PROPERTY_NAME = '__property_name__'
_LOGGED_MODEL__PROPERTIES = '__properties__'
_LOGGED_MODEL__PROPERTY_ATTR_PREFIX = '_property_'

_TimeType = TypeVar('_TimeType', bound=Union[float, int])
_ValueType = TypeVar('_ValueType')


@unique
class TimeMode(IntEnum):
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

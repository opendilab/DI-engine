from typing import List, Tuple, Callable, Any

from .base import ILoaderClass, Loader, CAPTURE_EXCEPTIONS
from .exception import CompositeStructureError
from .types import method
from .utils import raw

MAPPING_ERROR_ITEM = Tuple[str, Exception]
MAPPING_ERRORS = List[MAPPING_ERROR_ITEM]


class MappingError(CompositeStructureError):

    def __init__(self, key_errors: MAPPING_ERRORS, value_errors: MAPPING_ERRORS):
        self.__key_errors = list(key_errors or [])
        self.__value_errors = list(value_errors or [])
        self.__errors = self.__key_errors + self.__value_errors

    def key_errors(self) -> MAPPING_ERRORS:
        return self.__key_errors

    def value_errors(self) -> MAPPING_ERRORS:
        return self.__value_errors

    def errors(self) -> MAPPING_ERRORS:
        return self.__errors


def mapping(key_loader, value_loader, type_back: bool = True) -> ILoaderClass:
    key_loader = Loader(key_loader)
    value_loader = Loader(value_loader)

    def _load(value):
        _key_errors = []
        _value_errors = []
        _result = {}
        for key_, value_ in value.items():
            key_error, value_error = None, None
            key_result, value_result = None, None

            try:
                key_result = key_loader(key_)
            except CAPTURE_EXCEPTIONS as err:
                key_error = err

            try:
                value_result = value_loader(value_)
            except CAPTURE_EXCEPTIONS as err:
                value_error = err

            if not key_error and not value_error:
                _result[key_result] = value_result
            else:
                if key_error:
                    _key_errors.append((key_, key_error))
                if value_error:
                    _value_errors.append((key_, value_error))

        if not _key_errors and not _value_errors:
            if type_back:
                _result = type(value)(_result)
            return _result
        else:
            raise MappingError(_key_errors, _value_errors)

    return method('items') & Loader(_load)


def mpfilter(check: Callable[[Any, Any], bool], type_back: bool = True) -> ILoaderClass:

    def _load(value):
        _result = {key_: value_ for key_, value_ in value.items() if check(key_, value_)}

        if type_back:
            _result = type(value)(_result)
        return _result

    return method('items') & Loader(_load)


def mpkeys() -> ILoaderClass:
    return method('items') & method('keys') & Loader(lambda v: set(v.keys()))


def mpvalues() -> ILoaderClass:
    return method('items') & method('values') & Loader(lambda v: set(v.values()))


def mpitems() -> ILoaderClass:
    return method('items') & Loader(lambda v: set([(key, value) for key, value in v.items()]))


_INDEX_PRECHECK = method('__getitem__')


def item(key) -> ILoaderClass:
    return _INDEX_PRECHECK & Loader(
        (lambda v: key in v.keys(), lambda v: v[key], KeyError('key {key} not found'.format(key=repr(key))))
    )


def item_or(key, default) -> ILoaderClass:
    return _INDEX_PRECHECK & (item(key) | raw(default))

from typing import List, Tuple

from .base import ILoaderClass, Loader, CAPTURE_EXCEPTIONS, _func_check_gene
from .exception import CompositeStructureError

MAPPING_ERROR_ITEM = Tuple[str, Exception]
MAPPING_ERRORS = List[MAPPING_ERROR_ITEM]

_check_items = _func_check_gene('items')


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
        _check_items(value)

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

    return Loader(_load)

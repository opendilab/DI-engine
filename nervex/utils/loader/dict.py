from typing import Mapping

from .base import Loader, CAPTURE_EXCEPTIONS, ILoaderClass
from .exception import CompositeStructureError

DICT_ERRORS = Mapping[str, Exception]


class DictError(CompositeStructureError):

    def __init__(self, errors: DICT_ERRORS):
        self.__error = errors

    @property
    def errors(self) -> DICT_ERRORS:
        return self.__error


def dict_(**kwargs) -> ILoaderClass:
    kwargs = [(k, Loader(v)) for k, v in kwargs.items()]

    def _load(value):
        _errors = {}
        _results = {}

        for k, vl in kwargs:
            try:
                v = vl(value)
            except CAPTURE_EXCEPTIONS as err:
                _errors[k] = err
            else:
                _results[k] = v

        if not _errors:
            return _results
        else:
            raise DictError(_errors)

    return Loader(_load)

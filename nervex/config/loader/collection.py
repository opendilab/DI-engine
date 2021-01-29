from typing import Optional, List, Tuple

from .base import ILoaderClass, Loader, CAPTURE_EXCEPTIONS
from .exception import CompositeStructureError

COLLECTION_ERROR_ITEM = Tuple[int, Exception]
COLLECTION_ERRORS = List[COLLECTION_ERROR_ITEM]


class CollectionError(CompositeStructureError):

    def __init__(self, errors: COLLECTION_ERRORS):
        self.__errors = list(errors or [])
        CompositeStructureError.__init__(
            self, '{count} error(s) found in collection.'.format(count=repr(list(self.__errors)))
        )

    @property
    def errors(self) -> COLLECTION_ERRORS:
        return self.__errors


def collection(loader, type_back: bool = True) -> ILoaderClass:
    loader = Loader(loader)

    def _load(value):
        if hasattr(value, '__iter__'):
            _result = []
            _errors = []

            for index, item in enumerate(value):
                try:
                    _return = loader.load(item)
                except CAPTURE_EXCEPTIONS as err:
                    _errors.append((index, err))
                else:
                    _result.append(_return)

            if _errors:
                raise CollectionError(_errors)

            if type_back:
                _result = type(value)(_result)
            return _result
        else:
            raise TypeError('type {type} not support __iter__'.format(type=repr(type(value).__name__)))

    return Loader(_load)


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
        if hasattr(value, 'items'):
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
        else:
            raise TypeError('type {type} not support items'.format(type=repr(type(value).__name__)))

    return Loader(_load)


def tuple_(*loaders) -> ILoaderClass:
    loaders = [Loader(loader) for loader in loaders]

    def _load(value: tuple):
        return tuple([loader(item) for loader, item in zip(loaders, value)])

    return tuple & length_is(len(loaders)) & Loader(_load)


def length(min_length: Optional[int] = None, max_length: Optional[int] = None) -> ILoaderClass:

    def _load(value):
        if hasattr(value, '__len__'):
            _length = len(value)
            if min_length is not None and _length < min_length:
                raise ValueError(
                    'minimum length is {expect}, but {actual} found'.format(
                        expect=repr(min_length), actual=repr(_length)
                    )
                )
            if max_length is not None and _length > max_length:
                raise ValueError(
                    'maximum length is {expect}, but {actual} found'.format(
                        expect=repr(max_length), actual=repr(_length)
                    )
                )

            return value
        else:
            raise TypeError('type {type} not support __len__'.format(type=repr(value.__class__.__name__)))

    return Loader(_load)


def length_is(length_: int) -> ILoaderClass:
    return length(min_length=length_, max_length=length_)


def contains(content) -> ILoaderClass:

    def _load(value):
        if hasattr(value, '__contains__'):
            if content not in value:
                raise ValueError('{content} not found in value'.format(content=repr(content)))

            return value
        else:
            raise TypeError('type {type} not support __contains__'.format(type=repr(value.__class__.__name__)))

    return Loader(_load)

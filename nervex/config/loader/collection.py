from typing import Optional, List, Tuple, Callable, Any

from .base import ILoaderClass, Loader, CAPTURE_EXCEPTIONS, _func_check_gene
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


_check_iter = _func_check_gene('__iter__')
_check_len = _func_check_gene('__len__')
_check_contains = _func_check_gene('__contains__')


def collection(loader, type_back: bool = True) -> ILoaderClass:
    loader = Loader(loader)

    def _load(value):
        _check_iter(value)

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

    return Loader(_load)


def tuple_(*loaders) -> ILoaderClass:
    loaders = [Loader(loader) for loader in loaders]

    def _load(value: tuple):
        return tuple([loader(item) for loader, item in zip(loaders, value)])

    return tuple & length_is(len(loaders)) & Loader(_load)


def length(min_length: Optional[int] = None, max_length: Optional[int] = None) -> ILoaderClass:

    def _load(value):
        _check_len(value)

        _length = len(value)
        if min_length is not None and _length < min_length:
            raise ValueError(
                'minimum length is {expect}, but {actual} found'.format(expect=repr(min_length), actual=repr(_length))
            )
        if max_length is not None and _length > max_length:
            raise ValueError(
                'maximum length is {expect}, but {actual} found'.format(expect=repr(max_length), actual=repr(_length))
            )

        return value

    return Loader(_load)


def length_is(length_: int) -> ILoaderClass:
    return length(min_length=length_, max_length=length_)


def contains(content) -> ILoaderClass:

    def _load(value):
        _check_contains(value)

        if content not in value:
            raise ValueError('{content} not found in value'.format(content=repr(content)))

        return value

    return Loader(_load)


def cofilter(checker: Callable[[Any], bool], type_back: bool = True):

    def _load(value):
        _result = None
        if type_back:
            _result = type(value)(_result)
        return _result

    return Loader(_load)


def tpfilter(*indices):

    def _load(value: tuple):
        return (value[index] for index in indices)

    return tuple & Loader(_load)

from abc import abstractmethod
from typing import TypeVar, Callable, Any

CAPTURE_EXCEPTIONS = (Exception, )
_ValueType = TypeVar('_ValueType')


def _to_exception(exception) -> Callable[[Any], Exception]:
    if hasattr(exception, '__call__'):
        return exception
    elif isinstance(exception, Exception):
        return lambda v: exception
    elif isinstance(exception, str):
        return lambda v: ValueError(exception)
    else:
        raise TypeError(
            'Unknown type of exception, func, exception or str expected but {actual} found.'.format(
                actual=repr(type(exception).__name__)
            )
        )


def _to_loader(value) -> 'ILoaderClass':
    if isinstance(value, ILoaderClass):
        return value
    elif isinstance(value, tuple):
        if len(value) == 2:
            _predict, _exception = value
            _load = None
        elif len(value) == 3:
            _predict, _load, _exception = value
        else:
            raise ValueError('Tuple\'s length should be 2 or 3, but {actual} found.'.format(actual=repr(len(value))))

        _exception = _to_exception(_exception)

        def _load_tuple(value_):
            if not _predict(value_):
                raise _exception(value_)

            return (_load or (lambda v: v))(value_)

        return _to_loader(_load_tuple)
    elif isinstance(value, type):

        def _load_type(value_):
            if not isinstance(value_, value):
                raise TypeError(
                    'type not match, {expect} expected but {actual} found'.format(
                        expect=repr(value.__name__), actual=repr(type(value_).__name__)
                    )
                )
            return value_

        return _to_loader(_load_type)
    elif hasattr(value, '__call__'):

        class _Loader(ILoaderClass):

            def _load(self, value_):
                return value(value_)

        return _Loader()
    elif isinstance(value, bool):
        return _to_loader((lambda v: value, ValueError('assertion false')))
    elif value is None:
        return _to_loader(
            (
                lambda v: v is None, lambda v:
                TypeError('type not match, none expected but {actual} found'.format(actual=repr(type(v).__name__)))
            )
        )
    else:
        return _to_loader(lambda v: value)


Loader = _to_loader


def _reset_exception(loader, eg: Callable[[Any, Exception], Exception]):
    loader = Loader(loader)

    def _load(value):
        try:
            return loader(value)
        except CAPTURE_EXCEPTIONS as err:
            raise eg(value, err)

    return Loader(_load)


class ILoaderClass:

    @abstractmethod
    def _load(self, value: _ValueType) -> _ValueType:
        raise NotImplementedError

    def __load(self, value: _ValueType) -> _ValueType:
        return self._load(value)

    def __check(self, value: _ValueType) -> bool:
        try:
            self._load(value)
        except CAPTURE_EXCEPTIONS:
            return False
        else:
            return True

    def load(self, value: _ValueType) -> _ValueType:
        return self.__load(value)

    def check(self, value: _ValueType) -> bool:
        return self.__check(value)

    def __call__(self, value: _ValueType) -> _ValueType:
        return self.__load(value)

    def __and__(self, other) -> 'ILoaderClass':

        def _load(value: _ValueType) -> _ValueType:
            self.load(value)
            return Loader(other).load(value)

        return Loader(_load)

    def __rand__(self, other) -> 'ILoaderClass':
        return Loader(other) & self

    def __or__(self, other) -> 'ILoaderClass':

        def _load(value: _ValueType) -> _ValueType:
            try:
                return self.load(value)
            except CAPTURE_EXCEPTIONS:
                return Loader(other).load(value)

        return Loader(_load)

    def __ror__(self, other) -> 'ILoaderClass':
        return Loader(other) | self

    def __rshift__(self, other) -> 'ILoaderClass':

        def _load(value: _ValueType) -> _ValueType:
            _return_value = self.load(value)
            return _to_loader(other).load(_return_value)

        return Loader(_load)

    def __rrshift__(self, other) -> 'ILoaderClass':
        return Loader(other) >> self

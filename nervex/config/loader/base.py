from abc import abstractmethod
from typing import TypeVar

CAPTURE_EXCEPTIONS = (Exception,)
_ValueType = TypeVar('_ValueType')


def _to_loader(value) -> 'ILoaderClass':
    if isinstance(value, ILoaderClass):
        return value
    elif isinstance(value, type):
        def _load_type(value_):
            if not isinstance(value_, value):
                raise TypeError(
                    'type not match, {expect} expected but {actual} found'.format(expect=repr(value.__name__),
                                                                                  actual=repr(type(value_).__name__)))
            return value_

        return _to_loader(_load_type)
    elif hasattr(value, '__call__'):
        class _Loader(ILoaderClass):
            def _load(self, value_):
                return value(value_)

        return _Loader()
    elif isinstance(value, bool):
        def _load_bool(value_):
            if not value:
                raise ValueError('assertion false')
            return value_

        return _to_loader(_load_bool)
    elif value is None:
        def _load_none(value_):
            if value_ is not None:
                raise ValueError(
                    'type not match, none expected but {actual} found'.format(actual=repr(value_).__name__))
            return value_

        return _to_loader(_load_none)
    else:
        raise TypeError('Unknown type for loader.')


Loader = _to_loader


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

    def __call__(self, value: _ValueType) -> bool:
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

from typing import Optional

from .base import ILoaderClass, Loader


def collection(loader, type_back: bool = True) -> ILoaderClass:
    loader = Loader(loader)

    def _load(value):
        if hasattr(value, '__iter__'):
            _result = [loader.load(item) for item in value]
            if type_back:
                _result = type(value)(_result)
            return _result
        else:
            raise TypeError('type {type} not support __iter__'.format(type=repr(value.__class__.__name__)))

    return Loader(_load)


def mapping(key_loader, value_loader, type_back: bool = True) -> ILoaderClass:
    key_loader = Loader(key_loader)
    value_loader = Loader(value_loader)

    def _load(value):
        if hasattr(value, 'items'):
            _result = {key_loader(key_): value_loader(value_) for key_, value_ in value.items()}
            if type_back:
                _result = type(value)(_result)
            return _result
        else:
            raise TypeError()

    return Loader(_load)


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

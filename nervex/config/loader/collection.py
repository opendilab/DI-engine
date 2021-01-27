from typing import Optional

from .base import ILoaderClass, Loader


def collection(validator) -> ILoaderClass:
    validator = Loader(validator)

    def _load(value):
        if hasattr(value, '__iter__'):
            return [validator.load(item) for item in value]
        else:
            raise TypeError('type {type} not support __iter__'.format(type=repr(value.__class__.__name__)))

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

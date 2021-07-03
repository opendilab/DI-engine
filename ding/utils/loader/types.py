from functools import partial

from .base import Loader, ILoaderClass, _reset_exception
from .utils import check_only


def is_type(type_: type) -> ILoaderClass:
    if isinstance(type_, type):
        return Loader(type_)
    else:
        raise TypeError('Type variable expected but {actual} found.'.format(actual=repr(type(type_).__name__)))


def to_type(type_: type) -> ILoaderClass:
    return Loader(lambda v: type_(v))


def is_callable() -> ILoaderClass:
    return _reset_exception(
        check_only(prop('__call__')),
        lambda v, e: TypeError('callable expected but {func} not found'.format(func=repr('__call__')))
    )


def prop(attr_name: str) -> ILoaderClass:
    return Loader(
        (
            lambda v: hasattr(v, attr_name), lambda v: getattr(v, attr_name),
            AttributeError('attribute {name} expected but not found'.format(name=repr(attr_name)))
        )
    )


def method(method_name: str) -> ILoaderClass:
    return _reset_exception(
        prop(method_name) >> is_callable(), lambda v, e:
        TypeError('type {type} not support function {func}'.format(type=repr(type(v).__name__), func=repr('__iter__')))
    )


def fcall(*args, **kwargs) -> ILoaderClass:
    return Loader(lambda v: v(*args, **kwargs))


def fpartial(*args, **kwargs) -> ILoaderClass:
    return Loader(lambda v: partial(v, *args, **kwargs))

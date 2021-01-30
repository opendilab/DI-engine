from .base import Loader, ILoaderClass, _reset_exception


def is_type(type_: type) -> ILoaderClass:
    if isinstance(type_, type):
        return Loader(type_)
    else:
        raise TypeError('Type variable expected but {actual} found.'.format(actual=repr(type(type_).__name__)))


def to_type(type_: type) -> ILoaderClass:
    return Loader(lambda v: type_(v))


def prop(attr_name: str) -> ILoaderClass:
    return Loader(
        (
            lambda v: hasattr(v, attr_name), lambda v: getattr(v, attr_name),
            AttributeError('attribute {name} expected but not found'.format(name=repr(attr_name)))
        )
    )


def func(func_name: str) -> ILoaderClass:
    return _reset_exception(
        prop(func_name) >> prop('__call__'), lambda v, e:
        TypeError('type {type} not support function {func}'.format(type=repr(type(v).__name__), func=repr('__iter__')))
    )

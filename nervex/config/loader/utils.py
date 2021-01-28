from .base import Loader, ILoaderClass


def keep() -> ILoaderClass:
    return Loader(lambda v: v)


def is_type(type_: type) -> ILoaderClass:
    if isinstance(type_, type):
        return Loader(type_)
    else:
        raise TypeError('Type variable expected but {actual} found.'.format(actual=repr(type(type_).__name__)))


def to_type(type_: type) -> ILoaderClass:
    return Loader(lambda v: type_(v))


def optional(loader) -> ILoaderClass:
    return Loader(loader) | None


def check_only(loader) -> ILoaderClass:
    return Loader(loader) & keep()

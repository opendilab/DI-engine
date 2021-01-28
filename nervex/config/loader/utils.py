from .base import Loader, ILoaderClass


def keep() -> ILoaderClass:
    return Loader(lambda v: v)


def to_type(type_: type) -> ILoaderClass:
    return Loader(lambda v: type_(v))


def optional(loader) -> ILoaderClass:
    return Loader(loader) | None


def check_only(loader) -> ILoaderClass:
    return Loader(loader) & keep()

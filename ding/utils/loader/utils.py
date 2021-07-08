from .base import Loader, ILoaderClass


def keep() -> ILoaderClass:
    return Loader(lambda v: v)


def raw(value) -> ILoaderClass:
    return Loader(lambda v: value)


def optional(loader) -> ILoaderClass:
    return Loader(loader) | None


def check_only(loader) -> ILoaderClass:
    return Loader(loader) & keep()


def check(loader) -> ILoaderClass:
    return Loader(lambda x: Loader(loader).check(x))

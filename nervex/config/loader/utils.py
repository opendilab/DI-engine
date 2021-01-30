from .base import Loader, ILoaderClass


def keep() -> ILoaderClass:
    return Loader(lambda v: v)


def optional(loader) -> ILoaderClass:
    return Loader(loader) | None


def check_only(loader) -> ILoaderClass:
    return Loader(loader) & keep()

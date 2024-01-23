from .base import Loader, ILoaderClass


def keep() -> ILoaderClass:
    """
    Overview:
        Create a keep loader.
    """

    return Loader(lambda v: v)


def raw(value) -> ILoaderClass:
    """
    Overview:
        Create a raw loader.
    """

    return Loader(lambda v: value)


def optional(loader) -> ILoaderClass:
    """
    Overview:
        Create a optional loader.
    Arguments:
        - loader (:obj:`ILoaderClass`): The loader.
    """

    return Loader(loader) | None


def check_only(loader) -> ILoaderClass:
    """
    Overview:
        Create a check only loader.
    Arguments:
        - loader (:obj:`ILoaderClass`): The loader.
    """

    return Loader(loader) & keep()


def check(loader) -> ILoaderClass:
    """
    Overview:
        Create a check loader.
    Arguments:
        - loader (:obj:`ILoaderClass`): The loader.
    """

    return Loader(lambda x: Loader(loader).check(x))

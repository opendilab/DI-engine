import random
import string
from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Mapping, Any, Dict

_LENGTH_OF_RANDOM_TOKEN = 64


def random_token(length: Optional[int] = None) -> str:
    """
    Overview:
        Generate random hex token
    Arguments:
        - length (:obj:`Optional[int]`): Length of the random token (`None` means `64`)
    Returns:
        - token (:obj:`str`): Generated random token
    Example:
        >>> random_token()  # '4eAbd5218e3d0da5e7AAFcBF48Ea0Df2dadED1bdDF0B8724FdE1569AA78F24A7'
        >>> random_token(24)  # 'Cd1CdD98caAb8602ac6501aC'
    """
    return ''.join([random.choice(string.hexdigits) for _ in range(length or _LENGTH_OF_RANDOM_TOKEN)])


class ControllableContext(metaclass=ABCMeta):
    """
    Overview:
        Basic context-supported class structure
    Example:
        - Common usage

        >>> c = MyControllableContext()  # One of the superclasses if ControllableContext
        >>> c.start()
        >>> try:
        >>>     pass  # do anything you like
        >>> finally:
        >>>     c.close()

        - Use with keyword (the same as code above)

        >>> c = MyControllableContext()  # One of the superclasses if ControllableContext
        >>> with c as cc:   # cc is c, have the same id
        >>>     pass  # do anything you like
    """

    @abstractmethod
    def start(self):
        """
        Overview:
            Start the context
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def close(self):
        """
        Overview:
            Close the context
        """
        raise NotImplementedError  # pragma: no cover

    def __enter__(self):
        """
        Overview:
            Enter the context
        Returns:
            - self (:obj:`ControllableContext`): Context object itself
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Overview:
            Exit the context
        """
        self.close()


class ControllableService(ControllableContext, metaclass=ABCMeta):
    """
    Overview:
        Controllable service with context support, usually has concurrent feature.
    Example:
        - A common usage

        >>> c = MyControllableService()  # One of its superclasses is ControllableService
        >>> c.start()
        >>> try:
        >>>     pass  # do anything you like
        >>> finally:
        >>>     c.shutdown()  # shutdown the service
        >>>     c.join()  # wait until service is down

        - Use with keyword (the same as code above)

        >>> c = MyControllableService()  # One of its superclasses is ControllableService
        >>> with c as cc:   # cc is c, have the same id
        >>>     pass  # do anything you like
    """

    @abstractmethod
    def start(self):
        """
        Overview:
            Start the service
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def shutdown(self):
        """
        Overview:
            Shutdown the service (but service will not down immediately)
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def join(self):
        """
        Overview:
            Wait until the service is completely down
        """
        raise NotImplementedError  # pragma: no cover

    def close(self):
        """
        Overview:
            Close the service, wait until the service is down.
        """
        self.shutdown()
        self.join()


def translate_dict_func(d: Mapping[str, Callable[..., Any]]) -> Callable[..., Dict[str, Any]]:
    """
    Overview:
        Transform dict with funcs to function generating dict.
    Arguments:
        - d (:obj:`Mapping[str, Callable[..., Any]]`): Dict with funcs
    Returns:
        - func (:obj:`Callable[..., Dict[str, Any]]`): Function generating dict
    Example:
        >>> f1 = lambda x, y: x + y
        >>> f2 = lambda x, y: x - y
        >>> f3 = lambda x, y: x * y
        >>> fx = translate_dict_func({'a': f1, 'b': f2, 'c': f3})
        >>> fx(2, 3)  # {'a': 5, 'b': -1, 'c': 6}
        >>> fx(5, 11)  # ('a': 16, 'b': -6, 'c': 55}
    """

    def _func(*args, **kwargs) -> Dict[str, Any]:
        return {k: f(*args, **kwargs) for k, f in d.items()}

    return _func


def default_func(return_value=None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Overview:
        Transform optional function (maybe `None`) to function with default value
    Argument:
        - return_value (:obj:): Return value of the default function
    Returns:
        - decorator (:obj:`Callable[[Callable[..., Any]], Callable[..., Any]]`): A decorator function \
            that can decorator optional function to real function (must be not None)
    Example:
        >>> f1 = None
        >>> f2 = lambda x, y: x + y
        >>> ff1 = default_func()(f1)
        >>> ft1 = default_func(0)(f1)
        >>> ff2 = default_func()(f2)
        >>> ff1(2, 3)  # None
        >>> ft1(2, 3)  # 0
        >>> ff2(2, 3)  # 5
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # noinspection PyUnusedLocal
        def _func(*args, **kwargs):
            return return_value

        return func or _func

    return _decorator

import random
import string
from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Mapping, Any, Dict

_LENGTH_OF_RANDOM_TOKEN = 64


def random_token(length: Optional[int] = None) -> str:
    return ''.join([random.choice(string.hexdigits) for _ in range(length or _LENGTH_OF_RANDOM_TOKEN)])


class ControllableContext(metaclass=ABCMeta):

    @abstractmethod
    def start(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def close(self):
        raise NotImplementedError  # pragma: no cover

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ControllableService(ControllableContext, metaclass=ABCMeta):

    @abstractmethod
    def start(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def shutdown(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def join(self):
        raise NotImplementedError  # pragma: no cover

    def close(self):
        self.shutdown()
        self.join()


def translate_dict_func(d: Mapping[str, Callable[..., Any]]) -> Callable[..., Dict[str, Any]]:

    def _func(*args, **kwargs) -> Dict[str, Any]:
        return {k: f(*args, **kwargs) for k, f in d.items()}

    return _func


def default_func(return_value=None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # noinspection PyUnusedLocal
        def _func(*args, **kwargs):
            return return_value

        return func or _func

    return _decorator

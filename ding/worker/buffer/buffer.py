from typing import Any, Callable, List
from ding.worker.buffer.storage import Storage


def apply_middleware(func_name: str):

    def wrap_func(f: Callable):

        def _apply_middleware(buffer, *args):
            for func in buffer.middlewares:
                done, *args = func(buffer, func_name, *args)
                if done:
                    return args
            return f(buffer, *args)

        return _apply_middleware

    return wrap_func


class Buffer:

    def __init__(self, storage: Storage, **kwargs) -> None:
        self.storage = storage
        self.middlewares = []

    @apply_middleware("push")
    def push(self, data: Any) -> None:
        self.storage.append(data)

    @apply_middleware("sample")
    def sample(self, size: int) -> List[Any]:
        return self.storage.sample(size)

    @apply_middleware("clear")
    def clear(self) -> None:
        self.storage.clear()

    def use(self, func: Callable) -> "Buffer":
        r"""
        Overview:
            Use algorithm middlewares to modify the behavior of the buffer.
            Every middleware should be a callable function, it will receive three argument parts, including:
            1. The buffer instance, you can use this instance to visit every thing of the buffer, including the storage.
            2. The functions called by the user, there are three methods named `push`, `sample` and `clear`, so you can use these function name to decide which action to choose.
            3. The remaining arguments passed by the user to the original function, will be passed in *args.

            Each middleware handler should return two parts of the value, including:
            1. The first value is `done` (True or False), if done==True, the middleware chain will stop immediately, no more middlewares will be executed during this execution
            2. The remaining values, will be passed to the next middleware or the default function in the buffer.
        Arguments:
            - func (:obj:`Callable`): the middleware handler
        """
        self.middlewares.append(func)
        return self


class RateLimit:
    r"""
    Add rate limit threshold to push function
    """

    def __init__(self, max_rate: int = float("inf"), window_seconds: int = 30) -> None:
        self.max_rate = max_rate
        self.window_seconds = window_seconds
        self.buffered = []

    def handler(self) -> Callable:

        def _handler(buffer: Buffer, action: str, *args):
            if action == "push":
                return self.push(*args)
            return args

        return _handler

    def push(self, data) -> None:
        import time
        current = time.time()
        # Cut off stale records
        self.buffered = [t for t in self.buffered if t > current - self.window_seconds]
        if len(self.buffered) < self.max_rate:
            self.buffered.append(current)
            return False, data
        else:
            return True, None

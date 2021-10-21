from typing import Any, Callable, List
from ding.worker.buffer.storage import Storage
import copy


def apply_middleware(func_name: str):

    def wrap_func(base_func: Callable):

        def handler(buffer, *args, **kwargs):
            """
            The real processing starts here, we apply the middlewares one by one,
            each middleware will receive a `next` function, which is an executor of next
            middleware. You can change the input arguments to the `next` middleware, and you
            also can get the return value from the next middleware, so you have the
            maximum freedom to choose at what stage to implement your method.
            """

            def wrap_handler(middlewares, *args, **kwargs):
                if len(middlewares) == 0:
                    return base_func(buffer, *args, **kwargs)

                def next(*args, **kwargs):
                    return wrap_handler(middlewares[1:], *args, **kwargs)

                func = middlewares[0]
                return func(func_name, next, *args, **kwargs)

            return wrap_handler(buffer.middlewares, *args, **kwargs)

        return handler

    return wrap_func


class Buffer:

    def __init__(self, storage: Storage) -> None:
        """
        Overview:
            Initialize the buffer
        Arguments:
            - storage (:obj:`Storage`): The storage instance.
        """
        self.storage = storage
        self.middlewares = []

    @apply_middleware("push")
    def push(self, data: Any) -> None:
        r"""
        Overview:
            Push a data into buffer.
        Arguments:
            - data (:obj:`Any`): The data which will be pushed into buffer.
        """
        self.storage.append(data)

    @apply_middleware("sample")
    def sample(self, size: int, replace: bool = False, range: slice = None) -> List[Any]:
        """
        Overview:
            Sample data with length ``size``, this function may be wrapped by middlewares.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
        Returns:
            - sample_data (:obj:`list`): A list of data with length ``size``.
        """
        return self.storage.sample(size, replace=replace, range=range)

    @apply_middleware("clear")
    def clear(self) -> None:
        """
        Overview:
            Clear the storage
        """
        self.storage.clear()

    def use(self, func: Callable) -> "Buffer":
        r"""
        Overview:
            Use algorithm middlewares to modify the behavior of the buffer.
            Every middleware should be a callable function, it will receive three argument parts, including:
            1. The buffer instance, you can use this instance to visit every thing of the buffer,
               including the storage.
            2. The functions called by the user, there are three methods named `push`, `sample` and `clear`,
               so you can use these function name to decide which action to choose.
            3. The remaining arguments passed by the user to the original function, will be passed in *args.

            Each middleware handler should return two parts of the value, including:
            1. The first value is `done` (True or False), if done==True, the middleware chain will stop immediately,
               no more middlewares will be executed during this execution
            2. The remaining values, will be passed to the next middleware or the default function in the buffer.
        Arguments:
            - func (:obj:`Callable`): The middleware handler
        Returns:
            - buffer (:obj:`Buffer`): The instance self
        """
        self.middlewares.append(func)
        return self

    def view(self) -> "Buffer":
        r"""
        Overview:
            A view is a new instance of buffer, with a deepcopy of every property except the storage.
            The storage is shared among all the buffer instances.
        Returns:
            - buffer (:obj:`Buffer`): The instance self
        """
        buffer = Buffer(self.storage)
        buffer.middlewares = copy.deepcopy(self.middlewares)
        return buffer

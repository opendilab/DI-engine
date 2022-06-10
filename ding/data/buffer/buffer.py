from abc import abstractmethod, ABC
from typing import Any, List, Optional, Union, Callable
import copy
from dataclasses import dataclass
from functools import wraps
from ding.utils import fastcopy


def apply_middleware(func_name: str):

    def wrap_func(base_func: Callable):

        @wraps(base_func)
        def handler(buffer, *args, **kwargs):
            """
            Overview:
                The real processing starts here, we apply the middleware one by one,
                each middleware will receive next `chained` function, which is an executor of next
                middleware. You can change the input arguments to the next `chained` middleware, and you
                also can get the return value from the next middleware, so you have the
                maximum freedom to choose at what stage to implement your method.
            """

            def wrap_handler(middleware, *args, **kwargs):
                if len(middleware) == 0:
                    return base_func(buffer, *args, **kwargs)

                def chain(*args, **kwargs):
                    return wrap_handler(middleware[1:], *args, **kwargs)

                func = middleware[0]
                return func(func_name, chain, *args, **kwargs)

            return wrap_handler(buffer._middleware, *args, **kwargs)

        return handler

    return wrap_func


@dataclass
class BufferedData:
    data: Any
    index: str
    meta: dict


# Register new dispatcher on fastcopy to avoid circular references
def _copy_buffereddata(d: BufferedData) -> BufferedData:
    return BufferedData(data=fastcopy.copy(d.data), index=d.index, meta=fastcopy.copy(d.meta))


fastcopy.dispatch[BufferedData] = _copy_buffereddata


class Buffer(ABC):
    """
    Buffer is an abstraction of device storage, third-party services or data structures,
    For example, memory queue, sum-tree, redis, or di-store.
    """

    def __init__(self, size: int) -> None:
        self._middleware = []
        self.size = size

    @abstractmethod
    def push(self, data: Any, meta: Optional[dict] = None) -> BufferedData:
        """
        Overview:
            Push data and it's meta information in buffer.
        Arguments:
            - data (:obj:`Any`): The data which will be pushed into buffer.
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
        Returns:
            - buffered_data (:obj:`BufferedData`): The pushed data.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            sample_range: Optional[slice] = None,
            ignore_insufficient: bool = False,
            groupby: Optional[str] = None,
            unroll_len: Optional[int] = None
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        """
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`Optional[int]`): The number of the data that will be sampled.
            - indices (:obj:`Optional[List[str]]`): Sample with multiple indices.
            - replace (:obj:`bool`): If use replace is true, you may receive duplicated data from the buffer.
            - sample_range (:obj:`slice`): Sample range slice.
            - ignore_insufficient (:obj:`bool`): If ignore_insufficient is true, sampling more than buffer size
                with no repetition will not cause an exception.
            - groupby (:obj:`Optional[str]`): Groupby key in meta, i.e. groupby="episode"
            - unroll_len (:obj:`Optional[int]`): Number of consecutive frames within a group.
        Returns:
            - sample_data (:obj:`Union[List[BufferedData], List[List[BufferedData]]]`):
                A list of data with length ``size``, may be nested if groupby is set.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, index: str, data: Optional[Any] = None, meta: Optional[dict] = None) -> bool:
        """
        Overview:
            Update data and meta by index
        Arguments:
            - index (:obj:`str`): Index of data.
            - data (:obj:`any`): Pure data.
            - meta (:obj:`dict`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, index: str):
        """
        Overview:
            Delete one data sample by index
        Arguments:
            - index (:obj:`str`): Index
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, idx: int) -> BufferedData:
        """
        Overview:
            Get item by subscript index
        Arguments:
            - idx (:obj:`int`): Subscript index
        Returns:
            - buffered_data (:obj:`BufferedData`): Item from buffer
        """
        raise NotImplementedError

    def use(self, func: Callable) -> "Buffer":
        r"""
        Overview:
            Use algorithm middleware to modify the behavior of the buffer.
            Every middleware should be a callable function, it will receive three argument parts, including:
            1. The buffer instance, you can use this instance to visit every thing of the buffer,
               including the storage.
            2. The functions called by the user, there are three methods named `push`, `sample` and `clear`,
               so you can use these function name to decide which action to choose.
            3. The remaining arguments passed by the user to the original function, will be passed in *args.

            Each middleware handler should return two parts of the value, including:
            1. The first value is `done` (True or False), if done==True, the middleware chain will stop immediately,
               no more middleware will be executed during this execution
            2. The remaining values, will be passed to the next middleware or the default function in the buffer.
        Arguments:
            - func (:obj:`Callable`): The middleware handler
        Returns:
            - buffer (:obj:`Buffer`): The instance self
        """
        self._middleware.append(func)
        return self

    def view(self) -> "Buffer":
        r"""
        Overview:
            A view is a new instance of buffer, with a deepcopy of every property except the storage.
            The storage is shared among all the buffer instances.
        Returns:
            - buffer (:obj:`Buffer`): The instance self
        """
        return copy.copy(self)

    def __copy__(self) -> "Buffer":
        raise NotImplementedError

    def __len__(self) -> int:
        return self.count()

    def __getitem__(self, idx: int) -> BufferedData:
        return self.get(idx)

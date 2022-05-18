from typing import Iterable, Any, Optional, List
from collections.abc import Sequence
import numbers
import time
import copy
from threading import Thread
from queue import Queue

import numpy as np
import torch


def to_device(item: Any, device: str, ignore_keys: list = []) -> Any:
    """
    Overview:
        Transfer data to certain device
    Arguments:
        - item (:obj:`Any`): the item to be transferred
        - device (:obj:`str`): the device wanted
        - ignore_keys (:obj:`list`): the keys to be ignored in transfer, defalut set to empty
    Returns:
        - item (:obj:`Any`): the transferred item

    .. note:

        Now supports item type: :obj:`torch.nn.Module`, :obj:`torch.Tensor`, :obj:`Sequence`, \
            :obj:`dict`, :obj:`numbers.Integral`, :obj:`numbers.Real`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

    """
    if isinstance(item, torch.nn.Module):
        return item.to(device)
    elif isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, Sequence):
        if isinstance(item, str):
            return item
        else:
            return [to_device(t, device) for t in item]
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] = to_device(item[k], device)
        return new_item
    elif isinstance(item, numbers.Integral) or isinstance(item, numbers.Real):
        return item
    elif isinstance(item, np.ndarray) or isinstance(item, np.bool_):
        return item
    elif item is None or isinstance(item, str):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_dtype(item: Any, dtype: type) -> Any:
    r"""
    Overview:
        Change data to certain dtype
    Arguments:
        - item (:obj:`Any`): the item to be dtype changed
        - dtype (:obj:`type`): the type wanted
    Returns:
        - item (:obj:`object`): the dtype changed item

    .. note:

        Now supports item type: :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`
    """
    if isinstance(item, torch.Tensor):
        return item.to(dtype=dtype)
    elif isinstance(item, Sequence):
        return [to_dtype(t, dtype) for t in item]
    elif isinstance(item, dict):
        return {k: to_dtype(item[k], dtype) for k in item.keys()}
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_tensor(
        item: Any,
        dtype: Optional[torch.dtype] = None,
        ignore_keys: list = [],
        transform_scalar: bool = True
) -> torch.Tensor:
    r"""
    Overview:
        Change `numpy.ndarray`, sequence of scalars to torch.Tensor, and keep other data types unchanged.
    Arguments:
        - item (:obj:`Any`): the item to be changed
        - dtype (:obj:`type`): the type of wanted tensor
    Returns:
        - item (:obj:`torch.Tensor`): the change tensor

    .. note:

        Now supports item type: :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """

    def transform(d):
        if dtype is None:
            return torch.as_tensor(d)
        else:
            return torch.tensor(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            if k in ignore_keys:
                new_data[k] = v
            else:
                new_data[k] = to_tensor(v, dtype, ignore_keys, transform_scalar)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return []
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_tensor(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_tensor(t, dtype, ignore_keys, transform_scalar))
            return new_data
    elif isinstance(item, np.ndarray):
        if dtype is None:
            if item.dtype == np.float64:
                return torch.FloatTensor(item)
            else:
                return torch.from_numpy(item)
        else:
            return torch.from_numpy(item).to(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        if transform_scalar:
            if dtype is None:
                return torch.as_tensor(item)
            else:
                return torch.as_tensor(item).to(dtype)
        else:
            return item
    elif item is None:
        return None
    elif isinstance(item, torch.Tensor):
        if dtype is None:
            return item
        else:
            return item.to(dtype)
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_ndarray(item: Any, dtype: np.dtype = None) -> np.ndarray:
    r"""
    Overview:
        Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.
    Arguments:
        - item (:obj:`object`): the item to be changed
        - dtype (:obj:`type`): the type of wanted ndarray
    Returns:
        - item (:obj:`object`): the changed ndarray

    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """

    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_list(item: Any) -> list:
    r"""
    Overview:
        Transform `torch.Tensor`, `numpy.ndarray` to `list`, keep other data types unchanged
    Arguments:
        - item (:obj:`Any`): the item to be transformed
    Returns:
        - item (:obj:`list`): the list after transformation

    .. note::

        Now supports item type: :obj:`torch.Tensor`, :obj:`numpy.ndarray`, :obj:`dict`, :obj:`list`, \
        :obj:`tuple` and :obj:`None`
    """
    if item is None:
        return item
    elif isinstance(item, torch.Tensor):
        return item.tolist()
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, list) or isinstance(item, tuple):
        return [to_list(t) for t in item]
    elif isinstance(item, dict):
        return {k: to_list(v) for k, v in item.items()}
    elif np.isscalar(item):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def tensor_to_list(item):
    r"""
    Overview:
        Transform `torch.Tensor` to `list`, keep other data types unchanged
    Arguments:
        - item (:obj:`Any`): the item to be transformed
    Returns:
        - item (:obj:`list`): the list after transformation

    .. note::

        Now supports item type: :obj:`torch.Tensor`, :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """
    if item is None:
        return item
    elif isinstance(item, torch.Tensor):
        return item.tolist()
    elif isinstance(item, list) or isinstance(item, tuple):
        return [tensor_to_list(t) for t in item]
    elif isinstance(item, dict):
        return {k: tensor_to_list(v) for k, v in item.items()}
    elif np.isscalar(item):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def same_shape(data: list) -> bool:
    r"""
    Overview:
        Judge whether all data elements in a list have the same shape.
    Arguments:
        - data (:obj:`list`): the list of data
    Returns:
        - same (:obj:`bool`): whether the list of data all have the same shape
    """
    assert (isinstance(data, list))
    shapes = [t.shape for t in data]
    return len(set(shapes)) == 1


class LogDict(dict):
    '''
    Overview:
        Derived from ``dict``; Would transform ``torch.Tensor`` to ``list`` for convenient logging.
    '''

    def _transform(self, data):
        if isinstance(data, torch.Tensor):
            new_data = data.tolist()
        else:
            new_data = data
        return new_data

    def __setitem__(self, key, value):
        new_value = self._transform(value)
        super().__setitem__(key, new_value)

    def update(self, data):
        for k, v in data.items():
            self.__setitem__(k, v)


def build_log_buffer():
    r"""
    Overview:
        Builg log buffer, a subclass of dict, which can transform the input data into log format.
    Returns:
        - log_buffer (:obj:`LogDict`): Log buffer dict
    """
    return LogDict()


class CudaFetcher(object):
    """
    Overview:
        Fetch data from source, and transfer it to specified device.
    Interfaces:
        run, close
    """

    def __init__(self, data_source: Iterable, device: str, queue_size: int = 4, sleep: float = 0.1) -> None:
        self._source = data_source
        self._queue = Queue(maxsize=queue_size)
        self._stream = torch.cuda.Stream()
        self._producer_thread = Thread(target=self._producer, args=(), name='cuda_fetcher_producer')
        self._sleep = sleep
        self._device = device

    def __next__(self) -> Any:
        return self._queue.get()

    def run(self) -> None:
        """
        Overview:
            Start ``producer`` thread: Keep fetching data from source,
            change the device, and put into ``queue`` for request.
        """
        self._end_flag = False
        self._producer_thread.start()

    def close(self) -> None:
        """
        Overview:
            Stop ``producer`` thread by setting ``end_flag`` to ``True`` .
        """
        self._end_flag = True

    def _producer(self) -> None:
        with torch.cuda.stream(self._stream):
            while not self._end_flag:
                if self._queue.full():
                    time.sleep(self._sleep)
                else:
                    data = next(self._source)
                    data = to_device(data, self._device)
                    self._queue.put(data)


def get_tensor_data(data: Any) -> Any:
    """
    Overview:
        Get pure tensor data from the given data(without disturbing grad computation graph)
    """
    if isinstance(data, torch.Tensor):
        return data.data.clone()
    elif data is None:
        return None
    elif isinstance(data, Sequence):
        return [get_tensor_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: get_tensor_data(v) for k, v in data.items()}
    else:
        raise TypeError("not support type in get_tensor_data: {}".format(type(data)))


def unsqueeze(data: Any, dim: int = 0) -> Any:
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(dim)
    elif isinstance(data, Sequence):
        return [unsqueeze(d) for d in data]
    elif isinstance(data, dict):
        return {k: unsqueeze(v, 0) for k, v in data.items()}
    else:
        raise TypeError("not support type in unsqueeze: {}".format(type(data)))


def get_null_data(template: Any, num: int) -> List[Any]:
    ret = []
    for _ in range(num):
        data = copy.deepcopy(template)
        data['null'] = True
        data['done'] = True
        data['reward'].zero_()
        ret.append(data)
    return ret

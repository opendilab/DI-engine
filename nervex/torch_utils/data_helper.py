import numbers
from collections.abc import Sequence
from typing import Iterable, Any
import time
from threading import Thread
from queue import Queue

import numpy as np
import torch


def to_device(item, device, ignore_keys=[]):
    r"""
    Overview:
        transfer data to certain device

    Arguments:
        Note:
            Now supported item type :obj:`torch.nn.Module`, :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`,
            :obj:`numbers.Integral`, :obj:`numbers.Real`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

        - item (:obj:`object`): the item to be transfered
        - device (:obj:`torch.divice`): the device wanted
        - ignore_keys (:obj:`list` of `item.keys()`): the keys to be ignored in transfer, defalut set to empty

    Returns:
        - item (:obj:`object`): the transfered item
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


def to_dtype(item, dtype):
    r"""
    Overview:
        transfer data to certain dtype

    Arguments:
        Note:
            Now supported item type: :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`

        - item (:obj:`object`): the item to be transfered
        - dtype (:obj:`type`): the type wanted

    Returns:
        - item (:obj:`object`): the transfered item
    """
    if isinstance(item, torch.Tensor):
        return item.to(dtype=dtype)
    elif isinstance(item, Sequence):
        return [to_dtype(t, dtype) for t in item]
    elif isinstance(item, dict):
        return {k: to_dtype(item[k], dtype) for k in item.keys()}
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_tensor(item, dtype):
    r"""
    Overview:
        transfer data to certain dtype tensor

    Arguments:
        Note:
            Now supported item type: :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`

        - item (:obj:`object`): the item to be transfered
        - dtype (:obj:`type`): the type of wanted tensor

    Returns:
        - item (:obj:`object`): the transfered item
    """

    def transform(d):
        return torch.tensor(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_tensor(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        else:
            new_data = []
            for t in item:
                new_data.append(to_tensor(t, dtype))
            return new_data
    elif isinstance(item, np.ndarray):
        return torch.from_numpy(item).to(dtype)
    elif np.isscalar(item):
        return torch.as_tensor([item]).to(dtype)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def tensor_to_list(item):
    r"""
    Overview:
        transfer data to certain dtype

    Arguments:
        Note:
            Now supported item type: :obj:`torch.Tensor`, :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`

        - item (:obj:`object`): the item to be transfered

    Returns:
        - item (:obj:`list`): the transfered list
    """
    if item is None:
        return item
    elif isinstance(item, torch.Tensor):
        if item.shape == (1, ):
            return item.item()
        else:
            return item.tolist()
    elif isinstance(item, list) or isinstance(item, tuple):
        return [tensor_to_list(t) for t in item]
    elif isinstance(item, dict):
        return {k: tensor_to_list(v) for k, v in item.items()}
    elif np.isscalar(item):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def same_shape(data):
    r"""
    Overview:
        whether a list of data have same shapes

    Arguments:
        - data (:obj:`list`): the list of data

    Returns:
        - same (:obj:`bool`): whether the list of data all have same shapes
    """
    assert (isinstance(data, list))
    shapes = [t.shape for t in data]
    return len(set(shapes)) == 1


class LogDict(dict):

    def _transform(self, data):
        if isinstance(data, torch.Tensor):
            if data.shape == (1, ) or data.shape == ():
                new_data = data.item()
            else:
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
        builg log buffer, a subclass of dict which can transform the input data into log format
    Returns:
        - log_buffer (:obj:`LogDict`): log buffer dict
    """
    return LogDict()


class CudaFetcher(object):

    def __init__(self, data_source: Iterable, device: str, queue_size: int = 4, sleep: float = 0.1) -> None:
        self._source = data_source
        self._queue = Queue(maxsize=queue_size)
        self._stream = torch.cuda.Stream()
        self._producer_thread = Thread(target=self._producer, args=())
        self._sleep = sleep
        self._device = device

    def __next__(self) -> Any:
        return self._queue.get()

    def run(self) -> None:
        self._end_flag = False
        self._producer_thread.start()

    def close(self) -> None:
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
        get pure tensor data from the given data(avoiding disturbing grad computation graph)
    """
    if isinstance(data, torch.Tensor):
        return data.data.clone()
    elif data is None:
        return None
    elif isinstance(data, Sequence):
        return [get_tensor_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: v for k, v in data.items()}
    else:
        raise TypeError("not support type in get_tensor_data: {}".format(type(data)))

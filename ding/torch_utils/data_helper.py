from typing import Iterable, Any, Optional, List
from collections.abc import Sequence
import numbers
import time
import copy
from threading import Thread
from queue import Queue

import numpy as np
import torch
import treetensor.torch as ttorch

from ding.utils.default_helper import get_shape0


def to_device(item: Any, device: str, ignore_keys: list = []) -> Any:
    """
    Overview:
        Transfer data to certain device.
    Arguments:
        - item (:obj:`Any`): The item to be transferred.
        - device (:obj:`str`): The device wanted.
        - ignore_keys (:obj:`list`): The keys to be ignored in transfer, default set to empty.
    Returns:
        - item (:obj:`Any`): The transferred item.
    Examples:
        >>> setup_data_dict['module'] = nn.Linear(3, 5)
        >>> device = 'cuda'
        >>> cuda_d = to_device(setup_data_dict, device, ignore_keys=['module'])
        >>> assert cuda_d['module'].weight.device == torch.device('cpu')

    Examples:
        >>> setup_data_dict['module'] = nn.Linear(3, 5)
        >>> device = 'cuda'
        >>> cuda_d = to_device(setup_data_dict, device)
        >>> assert cuda_d['module'].weight.device == torch.device('cuda:0')

    .. note:

        Now supports item type: :obj:`torch.nn.Module`, :obj:`torch.Tensor`, :obj:`Sequence`, \
            :obj:`dict`, :obj:`numbers.Integral`, :obj:`numbers.Real`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

    """
    if isinstance(item, torch.nn.Module):
        return item.to(device)
    elif isinstance(item, ttorch.Tensor):
        if 'prev_state' in item:
            prev_state = to_device(item.prev_state, device)
            del item.prev_state
            item = item.to(device)
            item.prev_state = prev_state
            return item
        else:
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
    elif isinstance(item, torch.distributions.Distribution):  # for compatibility
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_dtype(item: Any, dtype: type) -> Any:
    """
    Overview:
        Change data to certain dtype.
    Arguments:
        - item (:obj:`Any`): The item for changing the dtype.
        - dtype (:obj:`type`): The type wanted.
    Returns:
        - item (:obj:`object`): The item with changed dtype.
    Examples (tensor):
        >>> t = torch.randint(0, 10, (3, 5))
        >>> tfloat = to_dtype(t, torch.float)
        >>> assert tfloat.dtype == torch.float

    Examples (list):
        >>> tlist = [torch.randint(0, 10, (3, 5))]
        >>> tlfloat = to_dtype(tlist, torch.float)
        >>> assert tlfloat[0].dtype == torch.float

    Examples (dict):
        >>> tdict = {'t': torch.randint(0, 10, (3, 5))}
        >>> tdictf = to_dtype(tdict, torch.float)
        >>> assert tdictf['t'].dtype == torch.float

    .. note:

        Now supports item type: :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`.
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
        item: Any, dtype: Optional[torch.dtype] = None, ignore_keys: list = [], transform_scalar: bool = True
) -> Any:
    """
    Overview:
        Convert ``numpy.ndarray`` object to ``torch.Tensor``.
    Arguments:
        - item (:obj:`Any`): The ``numpy.ndarray`` objects to be converted. It can be exactly a ``numpy.ndarray`` \
            object or a container (list, tuple or dict) that contains several ``numpy.ndarray`` objects.
        - dtype (:obj:`torch.dtype`): The type of wanted tensor. If set to ``None``, its dtype will be unchanged.
        - ignore_keys (:obj:`list`): If the ``item`` is a dict, values whose keys are in ``ignore_keys`` will not \
            be converted.
        - transform_scalar (:obj:`bool`): If set to ``True``, a scalar will be also converted to a tensor object.
    Returns:
        - item (:obj:`Any`): The converted tensors.

    Examples (scalar):
        >>> i = 10
        >>> t = to_tensor(i)
        >>> assert t.item() == i

    Examples (dict):
        >>> d = {'i': i}
        >>> dt = to_tensor(d, torch.int)
        >>> assert dt['i'].item() == i

    Examples (named tuple):
        >>> data_type = namedtuple('data_type', ['x', 'y'])
        >>> inputs = data_type(np.random.random(3), 4)
        >>> outputs = to_tensor(inputs, torch.float32)
        >>> assert type(outputs) == data_type
        >>> assert isinstance(outputs.x, torch.Tensor)
        >>> assert isinstance(outputs.y, torch.Tensor)
        >>> assert outputs.x.dtype == torch.float32
        >>> assert outputs.y.dtype == torch.float32

    .. note:

        Now supports item type: :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`.
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


def to_ndarray(item: Any, dtype: np.dtype = None) -> Any:
    """
    Overview:
        Convert ``torch.Tensor`` to ``numpy.ndarray``.
    Arguments:
        - item (:obj:`Any`): The ``torch.Tensor`` objects to be converted. It can be exactly a ``torch.Tensor`` \
            object or a container (list, tuple or dict) that contains several ``torch.Tensor`` objects.
        - dtype (:obj:`np.dtype`): The type of wanted array. If set to ``None``, its dtype will be unchanged.
    Returns:
        - item (:obj:`object`): The changed arrays.

    Examples (ndarray):
        >>> t = torch.randn(3, 5)
        >>> tarray1 = to_ndarray(t)
        >>> assert tarray1.shape == (3, 5)
        >>> assert isinstance(tarray1, np.ndarray)

    Examples (list):
        >>> t = [torch.randn(5, ) for i in range(3)]
        >>> tarray1 = to_ndarray(t, np.float32)
        >>> assert isinstance(tarray1, list)
        >>> assert tarray1[0].shape == (5, )
        >>> assert isinstance(tarray1[0], np.ndarray)

    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`.
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
        if dtype is None:
            return np.array(item)
        else:
            return np.array(item, dtype=dtype)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_list(item: Any) -> Any:
    """
    Overview:
        Convert ``torch.Tensor``, ``numpy.ndarray`` objects to ``list`` objects, and keep their dtypes unchanged.
    Arguments:
        - item (:obj:`Any`): The item to be converted.
    Returns:
        - item (:obj:`Any`): The list after conversion.

    Examples:
        >>> data = { \
                'tensor': torch.randn(4), \
                'list': [True, False, False], \
                'tuple': (4, 5, 6), \
                'bool': True, \
                'int': 10, \
                'float': 10., \
                'array': np.random.randn(4), \
                'str': "asdf", \
                'none': None, \
            } \
        >>> transformed_data = to_list(data)

    .. note::

        Now supports item type: :obj:`torch.Tensor`, :obj:`numpy.ndarray`, :obj:`dict`, :obj:`list`, \
        :obj:`tuple` and :obj:`None`.
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


def tensor_to_list(item: Any) -> Any:
    """
    Overview:
        Convert ``torch.Tensor`` objects to ``list``, and keep their dtypes unchanged.
    Arguments:
        - item (:obj:`Any`): The item to be converted.
    Returns:
        - item (:obj:`Any`): The lists after conversion.

    Examples (2d-tensor):
        >>> t = torch.randn(3, 5)
        >>> tlist1 = tensor_to_list(t)
        >>> assert len(tlist1) == 3
        >>> assert len(tlist1[0]) == 5

    Examples (1d-tensor):
        >>> t = torch.randn(3, )
        >>> tlist1 = tensor_to_list(t)
        >>> assert len(tlist1) == 3

    Examples (list)
        >>> t = [torch.randn(5, ) for i in range(3)]
        >>> tlist1 = tensor_to_list(t)
        >>> assert len(tlist1) == 3
        >>> assert len(tlist1[0]) == 5

    Examples (dict):
        >>> td = {'t': torch.randn(3, 5)}
        >>> tdlist1 = tensor_to_list(td)
        >>> assert len(tdlist1['t']) == 3
        >>> assert len(tdlist1['t'][0]) == 5

    .. note::

        Now supports item type: :obj:`torch.Tensor`, :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`.
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


def to_item(data: Any, ignore_error: bool = True) -> Any:
    """
    Overview:
        Convert data to python native scalar (i.e. data item), and keep their dtypes unchanged.
    Arguments:
        - data (:obj:`Any`): The data that needs to be converted.
        - ignore_error (:obj:`bool`): Whether to ignore the error when the data type is not supported. That is to \
            say, only the data can be transformed into a python native scalar will be returned.
    Returns:
        - data (:obj:`Any`): Converted data.

    Examples:
        >>>> data = { \
                'tensor': torch.randn(1), \
                'list': [True, False, torch.randn(1)], \
                'tuple': (4, 5, 6), \
                'bool': True, \
                'int': 10, \
                'float': 10., \
                'array': np.random.randn(1), \
                'str': "asdf", \
                'none': None, \
             }
        >>>> new_data = to_item(data)
        >>>> assert np.isscalar(new_data['tensor'])
        >>>> assert np.isscalar(new_data['array'])
        >>>> assert np.isscalar(new_data['list'][-1])

    .. note::

        Now supports item type: :obj:`torch.Tensor`, :obj:`torch.Tensor`, :obj:`ttorch.Tensor`, \
        :obj:`bool`, :obj:`str`, :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`.
    """
    if data is None:
        return data
    elif isinstance(data, bool) or isinstance(data, str):
        return data
    elif np.isscalar(data):
        return data
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor) or isinstance(data, ttorch.Tensor):
        return data.item()
    elif isinstance(data, list) or isinstance(data, tuple):
        return [to_item(d) for d in data]
    elif isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if ignore_error:
                try:
                    new_data[k] = to_item(v)
                except (ValueError, RuntimeError):
                    pass
            else:
                new_data[k] = to_item(v)
        return new_data
    else:
        raise TypeError("not support data type: {}".format(data))


def same_shape(data: list) -> bool:
    """
    Overview:
        Judge whether all data elements in a list have the same shapes.
    Arguments:
        - data (:obj:`list`): The list of data.
    Returns:
        - same (:obj:`bool`): Whether the list of data all have the same shape.

    Examples:
        >>> tlist = [torch.randn(3, 5) for i in range(5)]
        >>> assert same_shape(tlist)
        >>> tlist = [torch.randn(3, 5), torch.randn(4, 5)]
        >>> assert not same_shape(tlist)
    """
    assert (isinstance(data, list))
    shapes = [t.shape for t in data]
    return len(set(shapes)) == 1


class LogDict(dict):
    """
    Overview:
        Derived from ``dict``. Would convert ``torch.Tensor`` to ``list`` for convenient logging.
    Interfaces:
        ``_transform``, ``__setitem__``, ``update``.
    """

    def _transform(self, data: Any) -> None:
        """
        Overview:
            Convert tensor objects to lists for better logging.
        Arguments:
            - data (:obj:`Any`): The input data to be converted.
        """
        if isinstance(data, torch.Tensor):
            new_data = data.tolist()
        else:
            new_data = data
        return new_data

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Overview:
            Override the ``__setitem__`` function of built-in dict.
        Arguments:
            - key (:obj:`Any`): The key of the data item.
            - value (:obj:`Any`): The value of the data item.
        """
        new_value = self._transform(value)
        super().__setitem__(key, new_value)

    def update(self, data: dict) -> None:
        """
        Overview:
            Override the ``update`` function of built-in dict.
        Arguments:
            - data (:obj:`dict`): The dict for updating current object.
        """
        for k, v in data.items():
            self.__setitem__(k, v)


def build_log_buffer() -> LogDict:
    """
    Overview:
        Build log buffer, a subclass of dict, which can convert the input data into log format.
    Returns:
        - log_buffer (:obj:`LogDict`): Log buffer dict.
    Examples:
        >>> log_buffer = build_log_buffer()
        >>> log_buffer['not_tensor'] = torch.randn(3)
        >>> assert isinstance(log_buffer['not_tensor'], list)
        >>> assert len(log_buffer['not_tensor']) == 3
        >>> log_buffer.update({'not_tensor': 4, 'a': 5})
        >>> assert log_buffer['not_tensor'] == 4
    """
    return LogDict()


class CudaFetcher(object):
    """
    Overview:
        Fetch data from source, and transfer it to a specified device.
    Interfaces:
        ``__init__``, ``__next__``, ``run``, ``close``.
    """

    def __init__(self, data_source: Iterable, device: str, queue_size: int = 4, sleep: float = 0.1) -> None:
        """
        Overview:
            Initialize the CudaFetcher object using the given arguments.
        Arguments:
            - data_source (:obj:`Iterable`): The iterable data source.
            - device (:obj:`str`): The device to put data to, such as "cuda:0".
            - queue_size (:obj:`int`): The internal size of queue, such as 4.
            - sleep (:obj:`float`): Sleeping time when the internal queue is full.
        """
        self._source = data_source
        self._queue = Queue(maxsize=queue_size)
        self._stream = torch.cuda.Stream()
        self._producer_thread = Thread(target=self._producer, args=(), name='cuda_fetcher_producer')
        self._sleep = sleep
        self._device = device

    def __next__(self) -> Any:
        """
        Overview:
            Response to the request for data. Return one data item from the internal queue.
        Returns:
            - item (:obj:`Any`): The data item on the required device.
        """
        return self._queue.get()

    def run(self) -> None:
        """
        Overview:
            Start ``producer`` thread: Keep fetching data from source, change the device, and put into \
            ``queue`` for request.
        Examples:
            >>> timer = EasyTimer()
            >>> dataloader = iter([torch.randn(3, 3) for _ in range(10)])
            >>> dataloader = CudaFetcher(dataloader, device='cuda', sleep=0.1)
            >>> dataloader.run()
            >>> data = next(dataloader)
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
        """
        Overview:
            Keep fetching data from source, change the device, and put into ``queue`` for request.
        """

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
        Get pure tensor data from the given data (without disturbing grad computation graph).
    Arguments:
        - data (:obj:`Any`): The original data. It can be exactly a tensor or a container (Sequence or dict).
    Returns:
        - output (:obj:`Any`): The output data.
    Examples:
        >>> a = { \
                'tensor': torch.tensor([1, 2, 3.], requires_grad=True), \
                'list': [torch.tensor([1, 2, 3.], requires_grad=True) for _ in range(2)], \
                'none': None \
            }
        >>> tensor_a = get_tensor_data(a)
        >>> assert not tensor_a['tensor'].requires_grad
        >>> for t in tensor_a['list']:
        >>>     assert not t.requires_grad
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
    """
    Overview:
        Unsqueeze the tensor data.
    Arguments:
        - data (:obj:`Any`): The original data. It can be exactly a tensor or a container (Sequence or dict).
        - dim (:obj:`int`): The dimension to be unsqueezed.
    Returns:
        - output (:obj:`Any`): The output data.

    Examples (tensor):
        >>> t = torch.randn(3, 3)
        >>> tt = unsqueeze(t, dim=0)
        >>> assert tt.shape == torch.Shape([1, 3, 3])

    Examples (list):
        >>> t = [torch.randn(3, 3)]
        >>> tt = unsqueeze(t, dim=0)
        >>> assert tt[0].shape == torch.Shape([1, 3, 3])

    Examples (dict):
        >>> t = {"t": torch.randn(3, 3)}
        >>> tt = unsqueeze(t, dim=0)
        >>> assert tt["t"].shape == torch.Shape([1, 3, 3])
    """
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(dim)
    elif isinstance(data, Sequence):
        return [unsqueeze(d) for d in data]
    elif isinstance(data, dict):
        return {k: unsqueeze(v, 0) for k, v in data.items()}
    else:
        raise TypeError("not support type in unsqueeze: {}".format(type(data)))


def squeeze(data: Any, dim: int = 0) -> Any:
    """
    Overview:
        Squeeze the tensor data.
    Arguments:
        - data (:obj:`Any`): The original data. It can be exactly a tensor or a container (Sequence or dict).
        - dim (:obj:`int`): The dimension to be Squeezed.
    Returns:
        - output (:obj:`Any`): The output data.

    Examples (tensor):
        >>> t = torch.randn(1, 3, 3)
        >>> tt = squeeze(t, dim=0)
        >>> assert tt.shape == torch.Shape([3, 3])

    Examples (list):
        >>> t = [torch.randn(1, 3, 3)]
        >>> tt = squeeze(t, dim=0)
        >>> assert tt[0].shape == torch.Shape([3, 3])

    Examples (dict):
        >>> t = {"t": torch.randn(1, 3, 3)}
        >>> tt = squeeze(t, dim=0)
        >>> assert tt["t"].shape == torch.Shape([3, 3])
    """
    if isinstance(data, torch.Tensor):
        return data.squeeze(dim)
    elif isinstance(data, Sequence):
        return [squeeze(d) for d in data]
    elif isinstance(data, dict):
        return {k: squeeze(v, 0) for k, v in data.items()}
    else:
        raise TypeError("not support type in squeeze: {}".format(type(data)))


def get_null_data(template: Any, num: int) -> List[Any]:
    """
    Overview:
        Get null data given an input template.
    Arguments:
        - template (:obj:`Any`): The template data.
        - num (:obj:`int`): The number of null data items to generate.
    Returns:
        - output (:obj:`List[Any]`): The generated null data.

    Examples:
        >>> temp = {'obs': [1, 2, 3], 'action': 1, 'done': False, 'reward': torch.tensor(1.)}
        >>> null_data = get_null_data(temp, 2)
        >>> assert len(null_data) ==2
        >>> assert null_data[0]['null'] and null_data[0]['done']
    """
    ret = []
    for _ in range(num):
        data = copy.deepcopy(template)
        data['null'] = True
        data['done'] = True
        data['reward'].zero_()
        ret.append(data)
    return ret


def zeros_like(h: Any) -> Any:
    """
    Overview:
        Generate zero-tensors like the input data.
    Arguments:
        - h (:obj:`Any`): The original data. It can be exactly a tensor or a container (Sequence or dict).
    Returns:
        - output (:obj:`Any`): The output zero-tensors.

    Examples (tensor):
        >>> t = torch.randn(3, 3)
        >>> tt = zeros_like(t)
        >>> assert tt.shape == torch.Shape([3, 3])
        >>> assert torch.sum(torch.abs(tt)) < 1e-8

    Examples (list):
        >>> t = [torch.randn(3, 3)]
        >>> tt = zeros_like(t)
        >>> assert tt[0].shape == torch.Shape([3, 3])
        >>> assert torch.sum(torch.abs(tt[0])) < 1e-8

    Examples (dict):
        >>> t = {"t": torch.randn(3, 3)}
        >>> tt = zeros_like(t)
        >>> assert tt["t"].shape == torch.Shape([3, 3])
        >>> assert torch.sum(torch.abs(tt["t"])) < 1e-8
    """
    if isinstance(h, torch.Tensor):
        return torch.zeros_like(h)
    elif isinstance(h, (list, tuple)):
        return [zeros_like(t) for t in h]
    elif isinstance(h, dict):
        return {k: zeros_like(v) for k, v in h.items()}
    else:
        raise TypeError("not support type: {}".format(h))

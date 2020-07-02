import copy
from collections.abc import Sequence
import numbers
from itertools import product
from functools import reduce
from typing import Union, Any, Optional

import torch


class SequenceContainer:
    """
    Overview: Basic container that saves the data sequencely
    Interface: __init__, __len__, value, cat, __getitem__
    Property: name, keys
    """
    _name = 'SequenceContainer'

    def __init__(self, **kwargs):
        """
        Overview: init the container with input data(dict-style), and add the additional first dim for easy management
        Note: only supoort value type: torch.Tensor, Sequence
        """
        for k in kwargs.keys():
            if isinstance(kwargs[k], torch.Tensor):
                kwargs[k] = kwargs[k].unsqueeze(0)
            elif isinstance(kwargs[k], Sequence):
                kwargs[k] = [kwargs[k]]
            else:
                raise TypeError("not support type in class {}: {}".format(self._name, type(kwargs[k])))

        self.__dict__.update(kwargs)
        self._length = 1

    def __len__(self):
        """
        Overview: return the current length of the container
        Returns:
            - length (:obj:`int`): the value of the member variable _length
        """
        return self._length

    @property
    def name(self):
        """
        Overview: return the container class name
        Returns:
            - name (:obj:`str`): the value of the class variable _name
        Note: the subclass need to override the value of _name
        """
        return self._name

    @property
    def keys(self):
        """
        Overview: return the data keys
        Returns:
            - keys (:obj:`list`): the list including all the keys of the data
        """
        keys = list(self.__dict__.keys())
        keys.remove('_length')
        return keys

    def value(self, k):
        """
        Overview: get one of the value of all the elements in the container, according to input k
        Arguments:
            - k (:obj:`str`): the key to look up, must be in self.keys
        Returns:
            - value (:obj:`T`): one of the value, value type is specified by data
        """
        assert (k in self.keys)
        return self.__dict__[k]

    def cat(self, data):
        """
        Overview: concatenate the same class container object, inplace, each value does cat operation seperately
        Arguments:
            - data (:obj:`SequenceContainer`): the object need to be cat
        """
        assert (isinstance(data, SequenceContainer))
        assert (self.name == data.name)
        assert (self._length > 0)

        for k in data.keys:
            data_val = data.value(k)
            if data_val is None:
                continue
            if k not in self.keys or self.__dict__[k] is None:
                self.__dict__[k] = data_val
            elif isinstance(data_val, Sequence):
                self.__dict__[k] += data_val
            elif isinstance(data_val, torch.Tensor):
                self.__dict__[k] = torch.cat([self.__dict__[k], data_val])
            else:
                raise TypeError("not support type in class {}: {}".format(self._name, type(data_val)))
        self._length += len(data)

    def __getitem__(self, idx):
        # TODO(nyz) raw data interface
        """
        Overview: get data by the index, return the SequenceContainer object rather than raw data
        Arguments:
            - idx (:obj:`int`) the index of the container element
        Returns:
            - data (:obj:`SequenceContainer`) the element indicated by the index
        """
        data = {k: copy.deepcopy(self.__dict__[k][idx]) for k in self.keys}
        return SequenceContainer(**data)

    def __eq__(self, other):
        """
        Overview: judge whether the other object is equal to self
        Arguments:
            - other (:obj:`object`) the object need to be compared
        Returns:
            - eq_result (:obj:`bool`) whether the other object is equal to self
        """
        if not isinstance(other, SequenceContainer):
            return False
        if self.keys != other.keys:
            return False
        if len(self) != len(other):
            return False
        for k in self.keys:
            if isinstance(self.value(k), torch.Tensor):
                # for torch.Tensor:
                # (1) the same shape
                # (2) the same dtype
                # (3) the mean of the difference between two tensor is smaller than a tiny positive number
                if self.value(k).shape != other.value(k).shape:
                    return False
                if self.value(k).dtype != other.value(k).dtype:
                    return False
                if torch.abs(self.value(k) - other.value(k)).mean() > 1e-6:
                    return False
            else:
                if self.value(k) != other.value(k):
                    return False
        return True


class BaseContainer(object):
    """
    agent_num, trajectory_len, batch_size
    """
    AGENT_NUM_DIM = 0
    TRAJECTORY_LEN_DIM = 1
    BATCH_SIZE_DIM = 2

    def cat(self, other: 'BaseContainer', dim: int) -> None:
        raise NotImplementedError

    def __getitem__(self, idx: Union[int, slice, tuple, dict]) -> 'BaseContainer':
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    @property
    def data(self) -> list:
        raise NotImplementedError

    @property
    def shape(self) -> tuple:
        raise NotImplementedError

    @property
    def item(self) -> Any:
        raise NotImplementedError


class TensorContainer(BaseContainer):
    def __init__(self, data: torch.Tensor, shape: Optional[tuple] = tuple()) -> None:
        assert len(shape) == 0 or (len(shape) == 3 and all([isinstance(s, numbers.Integral) for s in shape]))
        if len(shape) == 0:
            self._data = data.view(1, 1, 1, *data.shape)
        else:
            assert data.shape[:3] == shape, 'the first three dim of data must match the input shape: {}/{}'.format(
                data.shape[:3], shape
            )
            self._data = data

    def cat(self, other: 'TensorContainer', dim: int) -> None:
        """
        Inplace cat
        """
        assert dim >= 0 and dim <= 2, "invalid dim value: {}".format(dim)
        assert all([self.shape[i] == other.shape[i]
                    for i in (set(range(3)) - set([dim]))]), '{}/{}'.format(self.shape, other.shape)
        assert self.item_shape == other.item_shape, '{}/{}'.format(self.item_shape, other.item_shape)
        self._data = torch.cat([self._data, other.data], dim=dim)

    def __getitem__(self, idx: Union[int, slice, tuple, dict]) -> 'TensorContainer':
        if isinstance(idx, slice):
            data = self._data[idx]
        elif isinstance(idx, int):
            data = self._data[slice(idx, idx + 1)]
        elif isinstance(idx, tuple):
            assert len(idx) <= 3
            keep_dim_idx = list(idx)
            for i, item in enumerate(keep_dim_idx):
                if isinstance(item, int):
                    tmp = keep_dim_idx[i]
                    keep_dim_idx[i] = slice(tmp, tmp + 1)
            data = self._data[keep_dim_idx]
        elif isinstance(idx, dict):
            selected_indexes = [None for _ in range(3)]
            for k, v in idx.items():
                dim = getattr(self, (k + '_dim').upper())
                selected_indexes[dim] = v
            handle = self._data
            for d, i in enumerate(selected_indexes):
                if i is None:
                    handle = handle
                else:
                    handle = torch.index_select(handle, d, torch.LongTensor(i))
            data = handle
        else:
            raise TypeError(type(idx))
        return TensorContainer(data, shape=data.shape[:3])

    def __repr__(self) -> str:
        return 'TensorContainer(agent_num={}, trajectory_len={}, batch_size={})'.format(*self.shape)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def shape(self) -> tuple:
        return tuple(self._data.shape[:3])

    @property
    def item(self) -> torch.Tensor:
        assert self.shape == (1, 1, 1), self.shape
        return self._data[0, 0, 0]

    @property
    def item_shape(self) -> tuple:
        return tuple(self._data[0, 0, 0].shape)


class SpecialContainer(BaseContainer):
    def __init__(self, data: Any, shape: Optional[tuple] = tuple()) -> None:
        self._data = []
        self._shape = []
        self._index_map = {}
        self._data_idx = 0

        if len(shape) == 0:
            self._data.append(data)
            self._shape = [1, 1, 1]
            self._index_map[self._get_index_key((0, 0, 0))] = 0
            self._data_idx += 1
        else:
            assert len(shape) == 3
            self._shape = list(shape)
            self._data = data
            indexes = product(*[range(i) for i in shape])
            for idx, index in enumerate(indexes):
                self._index_map[self._get_index_key(index)] = idx
            self._data_idx += reduce(lambda x, y: x * y, shape)

    def cat(self, other: 'SpecialContainer', dim: int) -> None:
        """
        Inplace cat
        """
        assert dim >= 0 and dim <= 2, "invalid dim value: {}".format(dim)
        assert all([self.shape[i] == other.shape[i]
                    for i in (set(range(3)) - set([dim]))]), '{}/{}'.format(self.shape, other.shape)

        self._data.extend(other.data)
        indexes = [range(i) for i in self.shape]
        indexes[dim] = [self.shape[dim] + i for i in range(other.shape[dim])]
        indexes = product(*indexes)
        for idx, index in enumerate(indexes):
            self._index_map[self._get_index_key(index)] = idx + self._data_idx
        self._data_idx += reduce(lambda x, y: x * y, other.shape)
        self._shape[dim] += other.shape[dim]

    def __getitem__(self, idx: Union[int, slice, tuple, dict]) -> 'SpecialContainer':
        if isinstance(idx, slice) or isinstance(idx, numbers.Integral):
            idx = tuple([idx])
        assert isinstance(idx, tuple) or isinstance(idx, dict), type(idx)

        selected_ranges = [None, None, None]
        if isinstance(idx, tuple):
            assert len(idx) <= 3
            for i, item in enumerate(idx):
                if isinstance(item, slice):
                    start = item.start if item.start else 0
                    stop = item.stop if item.stop else self.shape[i]
                    step = item.step if item.step else 1
                    assert start >= 0 and stop >= 0 and stop <= self.shape[i], '{}/{}'.format(start, stop)
                    assert start < stop, '{}/{}'.format(start, stop)
                    selected_ranges[i] = list(range(start, stop, step))
                elif isinstance(item, numbers.Integral):
                    assert item >= 0 and item < self.shape[i]
                    selected_ranges[i] = [item]
                else:
                    raise TypeError(type(item))
        elif isinstance(idx, dict):
            for k, v in idx.items():
                assert isinstance(v, list), type(v)
                dim = getattr(self, (k + '_dim').upper())
                selected_ranges[dim] = v
        for i, item in enumerate(selected_ranges):
            if item is None:
                selected_ranges[i] = list(range(self.shape[i]))

        selected_shape = tuple([len(t) for t in selected_ranges])
        selected_keys = [self._get_index_key(i) for i in product(*selected_ranges)]
        selected_indexes = [self._index_map[key] for key in selected_keys]
        selected_data = [self._data[i] for i in selected_indexes]
        return SpecialContainer(data=selected_data, shape=selected_shape)

    def __repr__(self) -> str:
        return 'SpecialContainer(agent_num={}, trajectory_len={}, batch_size={})'.format(*self.shape)

    @property
    def data(self) -> list:
        return self._data

    @property
    def shape(self) -> tuple:
        return tuple(self._shape)

    @property
    def item(self) -> Any:
        assert self.shape == (1, 1, 1), self.shape
        return self._data[0]

    def _get_index_key(self, index: tuple) -> str:
        return ''.join([str(i) for i in index])

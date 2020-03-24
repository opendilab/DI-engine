import torch
import copy
from collections import Sequence


class SequenceContainer(object):
    _name = 'SequenceContainer'

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._length = 1

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    @property
    def keys(self):
        keys = list(self.__dict__.keys())
        keys.remove('_length')
        return keys

    def value(self, k):
        assert (k in self.keys)
        return self.__dict__[k]

    def cat(self, data):
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
        data = {copy.deepcopy(self.__dict__[k]) for k in self.keys}
        return SequenceContainer(**data)

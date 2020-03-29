import copy
from collections.abc import Sequence

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

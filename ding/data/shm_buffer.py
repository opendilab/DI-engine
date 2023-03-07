from typing import Any, Optional, Union, Tuple, Dict
from multiprocessing import Array
import ctypes
import numpy as np
import torch

_NTYPE_TO_CTYPE = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class ShmBuffer():
    """
    Overview:
        Shared memory buffer to store numpy array.
    """

    def __init__(
            self,
            dtype: Union[type, np.dtype],
            shape: Tuple[int],
            copy_on_get: bool = True,
            ctype: Optional[type] = None
    ) -> None:
        """
        Overview:
            Initialize the buffer.
        Arguments:
            - dtype (:obj:`Union[type, np.dtype]`): The dtype of the data to limit the size of the buffer.
            - shape (:obj:`Tuple[int]`): The shape of the data to limit the size of the buffer.
            - copy_on_get (:obj:`bool`): Whether to copy data when calling get method.
            - ctype (:obj:`Optional[type]`): Origin class type, e.g. np.ndarray, torch.Tensor.
        """
        if isinstance(dtype, np.dtype):  # it is type of gym.spaces.dtype
            dtype = dtype.type
        self.buffer = Array(_NTYPE_TO_CTYPE[dtype], int(np.prod(shape)))
        self.dtype = dtype
        self.shape = shape
        self.copy_on_get = copy_on_get
        self.ctype = ctype

    def fill(self, src_arr: np.ndarray) -> None:
        """
        Overview:
            Fill the shared memory buffer with a numpy array. (Replace the original one.)
        Arguments:
            - src_arr (:obj:`np.ndarray`): array to fill the buffer.
        """
        assert isinstance(src_arr, np.ndarray), type(src_arr)
        # for np.array with shape (4, 84, 84) and float32 dtype, reshape is 15~20x faster than flatten
        # for np.array with shape (4, 84, 84) and uint8 dtype, reshape is 5~7x faster than flatten
        # so we reshape dst_arr rather than flatten src_arr
        dst_arr = np.frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)
        np.copyto(dst_arr, src_arr)

    def get(self) -> np.ndarray:
        """
        Overview:
            Get the array stored in the buffer.
        Return:
            - data (:obj:`np.ndarray`): A copy of the data stored in the buffer.
        """
        data = np.frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)
        if self.copy_on_get:
            data = data.copy()  # must use np.copy, torch.from_numpy and torch.as_tensor still use the same memory
        if self.ctype is torch.Tensor:
            data = torch.from_numpy(data)
        return data


class ShmBufferContainer(object):
    """
    Overview:
        Support multiple shared memory buffers. Each key-value is name-buffer.
    """

    def __init__(
            self,
            dtype: Union[Dict[Any, type], type, np.dtype],
            shape: Union[Dict[Any, tuple], tuple],
            copy_on_get: bool = True
    ) -> None:
        """
        Overview:
            Initialize the buffer container.
        Arguments:
            - dtype (:obj:`Union[type, np.dtype]`): The dtype of the data to limit the size of the buffer.
            - shape (:obj:`Union[Dict[Any, tuple], tuple]`): If `Dict[Any, tuple]`, use a dict to manage \
                multiple buffers; If `tuple`, use single buffer.
            - copy_on_get (:obj:`bool`): Whether to copy data when calling get method.
        """
        if isinstance(shape, dict):
            self._data = {k: ShmBufferContainer(dtype[k], v, copy_on_get) for k, v in shape.items()}
        elif isinstance(shape, (tuple, list)):
            self._data = ShmBuffer(dtype, shape, copy_on_get)
        else:
            raise RuntimeError("not support shape: {}".format(shape))
        self._shape = shape

    def fill(self, src_arr: Union[Dict[Any, np.ndarray], np.ndarray]) -> None:
        """
        Overview:
            Fill the one or many shared memory buffer.
        Arguments:
            - src_arr (:obj:`Union[Dict[Any, np.ndarray], np.ndarray]`): array to fill the buffer.
        """
        if isinstance(self._shape, dict):
            for k in self._shape.keys():
                self._data[k].fill(src_arr[k])
        elif isinstance(self._shape, (tuple, list)):
            self._data.fill(src_arr)

    def get(self) -> Union[Dict[Any, np.ndarray], np.ndarray]:
        """
        Overview:
            Get the one or many arrays stored in the buffer.
        Return:
            - data (:obj:`np.ndarray`): The array(s) stored in the buffer.
        """
        if isinstance(self._shape, dict):
            return {k: self._data[k].get() for k in self._shape.keys()}
        elif isinstance(self._shape, (tuple, list)):
            return self._data.get()

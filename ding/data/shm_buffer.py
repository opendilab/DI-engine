from typing import Any, Optional, Union, Tuple, Dict
from multiprocessing import Array
import ctypes
import numpy as np
import torch
import torch.multiprocessing as mp
from functools import reduce
from ditk import logging
from abc import abstractmethod

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

# uint16, uint32, uint32
_NTYPE_TO_TTYPE = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    # np.uint16: torch.int16,
    # np.uint32: torch.int32,
    # np.uint64: torch.int64,
    np.int8: torch.uint8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float32: torch.float32,
    np.float64: torch.float64,
}

_NOT_SUPPORT_NTYPE = {np.uint16: torch.int16, np.uint32: torch.int32, np.uint64: torch.int64}
_CONVERSION_TYPE = {np.uint16: np.int16, np.uint32: np.int32, np.uint64: np.int64}


class ShmBufferBase:

    @abstractmethod
    def fill(self, src_arr: Union[np.ndarray, torch.Tensor]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError


class ShmBuffer(ShmBufferBase):
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


class ShmBufferCuda(ShmBufferBase):

    def __init__(
            self,
            dtype: Union[torch.dtype, np.dtype],
            shape: Tuple[int],
            ctype: Optional[type] = None,
            copy_on_get: bool = True,
            device: Optional[torch.device] = torch.device('cuda:0')
    ) -> None:
        """
        Overview:
            Use torch.multiprocessing for shared tensor or ndaray between processes.
        Arguments:
            - dtype (Union[torch.dtype, np.dtype]): dtype of torch.tensor or numpy.ndarray.
            - shape (Tuple[int]): Shape of torch.tensor or numpy.ndarray.
            - ctype (type): Origin class type, e.g. np.ndarray, torch.Tensor.
            - copy_on_get (bool, optional): Can be set to False only if the shared object
                is a tenor, otherwise True.
            - device (Optional[torch.device], optional): The GPU device where cuda-shared-tensor
                is located, the default is cuda:0.

        Raises:
            RuntimeError: Unsupported share type by ShmBufferCuda.
        """
        if isinstance(dtype, np.dtype):  # it is type of gym.spaces.dtype
            self.ctype = np.ndarray
            dtype = dtype.type
            if dtype in _NOT_SUPPORT_NTYPE.keys():
                logging.warning(
                    "Torch tensor unsupport numpy type {}, attempt to do a type conversion, which may lose precision.".
                    format(dtype)
                )
                ttype = _NOT_SUPPORT_NTYPE[dtype]
                self.dtype = _CONVERSION_TYPE[dtype]
            else:
                ttype = _NTYPE_TO_TTYPE[dtype]
                self.dtype = dtype
        elif isinstance(dtype, torch.dtype):
            self.ctype = torch.Tensor
            ttype = dtype
        else:
            raise RuntimeError("The dtype parameter only supports torch.dtype and np.dtype")

        self.copy_on_get = copy_on_get
        self.shape = shape
        self.device = device
        self.buffer = torch.zeros(reduce(lambda x, y: x * y, shape), dtype=ttype, device=self.device)

    def fill(self, src_arr: Union[np.ndarray, torch.Tensor]) -> None:
        if self.ctype is np.ndarray:
            if src_arr.dtype.type != self.dtype:
                logging.warning(
                    "Torch tensor unsupport numpy type {}, attempt to do a type conversion, which may lose precision.".
                    format(self.dtype)
                )
                src_arr = src_arr.astype(self.dtype)
            tensor = torch.from_numpy(src_arr)
        elif self.ctype is torch.Tensor:
            tensor = src_arr
        else:
            raise RuntimeError("Unsopport CUDA-shared-tensor input type:\"{}\"".format(type(src_arr)))

        # If the GPU-a and GPU-b are connected using nvlink, the copy is very fast.
        with torch.no_grad():
            self.buffer.copy_(tensor.view(tensor.numel()))

    def get(self) -> Union[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            if self.ctype is np.ndarray:
                # Because ShmBufferCuda use CUDA memory exchanging data between processes.
                # So copy_on_get is necessary for numpy arrays.
                re = self.buffer.cpu()
                re = re.detach().view(self.shape).numpy()
            else:
                if self.copy_on_get:
                    re = self.buffer.clone().detach().view(self.shape)
                else:
                    re = self.buffer.view(self.shape)

        return re

    def __del__(self):
        del self.buffer


class ShmBufferContainer(object):
    """
    Overview:
        Support multiple shared memory buffers. Each key-value is name-buffer.
    """

    def __init__(
            self,
            dtype: Union[Dict[Any, type], type, np.dtype],
            shape: Union[Dict[Any, tuple], tuple],
            copy_on_get: bool = True,
            is_cuda_buffer: bool = False
    ) -> None:
        """
        Overview:
            Initialize the buffer container.
        Arguments:
            - dtype (:obj:`Union[type, np.dtype]`): The dtype of the data to limit the size of the buffer.
            - shape (:obj:`Union[Dict[Any, tuple], tuple]`): If `Dict[Any, tuple]`, use a dict to manage \
                multiple buffers; If `tuple`, use single buffer.
            - copy_on_get (:obj:`bool`): Whether to copy data when calling get method.
            - is_cuda_buffer (:obj:`bool`): Whether to use pytorch CUDA shared tensor as the implementation of shm.
        """
        if isinstance(shape, dict):
            self._data = {k: ShmBufferContainer(dtype[k], v, copy_on_get, is_cuda_buffer) for k, v in shape.items()}
        elif isinstance(shape, (tuple, list)):
            if not is_cuda_buffer:
                self._data = ShmBuffer(dtype, shape, copy_on_get)
            else:
                self._data = ShmBufferCuda(dtype, shape, copy_on_get)
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

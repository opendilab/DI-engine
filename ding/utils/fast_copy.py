import torch
import numpy as np
from typing import Any, List


class _FastCopy:
    """
    Overview:
        The idea of this class comes from this article \
        https://newbedev.com/what-is-a-fast-pythonic-way-to-deepcopy-just-data-from-a-python-dict-or-list.
        We use recursive calls to copy each object that needs to be copied, which will be 5x faster \
        than copy.deepcopy.
    Interfaces:
        ``__init__``, ``_copy_list``, ``_copy_dict``, ``_copy_tensor``, ``_copy_ndarray``, ``copy``.
    """

    def __init__(self):
        """
        Overview:
            Initialize the _FastCopy object.
        """

        dispatch = {}
        dispatch[list] = self._copy_list
        dispatch[dict] = self._copy_dict
        dispatch[torch.Tensor] = self._copy_tensor
        dispatch[np.ndarray] = self._copy_ndarray
        self.dispatch = dispatch

    def _copy_list(self, l: List) -> dict:
        """
        Overview:
            Copy the list.
        Arguments:
            - l (:obj:`List`): The list to be copied.
        """

        ret = l.copy()
        for idx, item in enumerate(ret):
            cp = self.dispatch.get(type(item))
            if cp is not None:
                ret[idx] = cp(item)
        return ret

    def _copy_dict(self, d: dict) -> dict:
        """
        Overview:
            Copy the dict.
        Arguments:
            - d (:obj:`dict`): The dict to be copied.
        """

        ret = d.copy()
        for key, value in ret.items():
            cp = self.dispatch.get(type(value))
            if cp is not None:
                ret[key] = cp(value)

        return ret

    def _copy_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Copy the tensor.
        Arguments:
            - t (:obj:`torch.Tensor`): The tensor to be copied.
        """

        return t.clone()

    def _copy_ndarray(self, a: np.ndarray) -> np.ndarray:
        """
        Overview:
            Copy the ndarray.
        Arguments:
            - a (:obj:`np.ndarray`): The ndarray to be copied.
        """

        return np.copy(a)

    def copy(self, sth: Any) -> Any:
        """
        Overview:
            Copy the object.
        Arguments:
            - sth (:obj:`Any`): The object to be copied.
        """

        cp = self.dispatch.get(type(sth))
        if cp is None:
            return sth
        else:
            return cp(sth)


fastcopy = _FastCopy()

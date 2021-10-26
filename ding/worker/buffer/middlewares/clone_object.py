from typing import Callable, Any, List
import torch
import numpy as np


class FastCopy:
    """
    The idea of this class comes from this article
    https://newbedev.com/what-is-a-fast-pythonic-way-to-deepcopy-just-data-from-a-python-dict-or-list.
    We use recursive calls to copy each object that needs to be copied, which will be 5x faster
    than copy.deepcopy.
    """

    def __init__(self):
        dispatch = {}
        dispatch[list] = self._copy_list
        dispatch[dict] = self._copy_dict
        dispatch[torch.Tensor] = self._copy_tensor
        dispatch[np.ndarray] = self._copy_ndarray
        self.dispatch = dispatch

    def _copy_list(self, l: List) -> dict:
        ret = l.copy()
        for idx, item in enumerate(ret):
            cp = self.dispatch.get(type(item))
            if cp is not None:
                ret[idx] = cp(item)
        return ret

    def _copy_dict(self, d: dict) -> dict:
        ret = d.copy()
        for key, value in ret.items():
            cp = self.dispatch.get(type(value))
            if cp is not None:
                ret[key] = cp(value)

        return ret

    def _copy_tensor(self, t: torch.Tensor) -> torch.Tensor:
        return t.clone()

    def _copy_ndarray(self, a: np.ndarray) -> np.ndarray:
        return np.copy(a)

    def copy(self, sth: Any) -> Any:
        cp = self.dispatch.get(type(sth))
        if cp is None:
            return sth
        else:
            return cp(sth)


def clone_object():
    """
    This middleware freezes the objects saved in memory buffer as a copy,
    try this middleware when you need to keep the object unchanged in buffer, and modify
    the object after sampling it (usuallly in multiple threads)
    """
    fastcopy = FastCopy()

    def push(next: Callable, data: Any, *args, **kwargs) -> None:
        data = fastcopy.copy(data)
        return next(data, *args, **kwargs)

    def sample(next: Callable, *args, **kwargs) -> List[Any]:
        data = next(*args, **kwargs)
        return fastcopy.copy(data)

    def _immutable_object(action: str, next: Callable, *args, **kwargs):
        if action == "push":
            return push(next, *args, **kwargs)
        elif action == "sample":
            return sample(next, *args, **kwargs)
        return next(*args, **kwargs)

    return _immutable_object

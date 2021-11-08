from typing import Callable, Any, List
from ding.worker.buffer import BufferedData
from ding.worker.buffer.utils import fastcopy


def clone_object():
    """
    This middleware freezes the objects saved in memory buffer as a copy,
    try this middleware when you need to keep the object unchanged in buffer, and modify
    the object after sampling it (usuallly in multiple threads)
    """

    def push(chain: Callable, data: Any, *args, **kwargs) -> BufferedData:
        data = fastcopy.copy(data)
        return chain(data, *args, **kwargs)

    def sample(chain: Callable, *args, **kwargs) -> List[BufferedData]:
        data = chain(*args, **kwargs)
        return fastcopy.copy(data)

    def _immutable_object(action: str, chain: Callable, *args, **kwargs):
        if action == "push":
            return push(chain, *args, **kwargs)
        elif action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _immutable_object

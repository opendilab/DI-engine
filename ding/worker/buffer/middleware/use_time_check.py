from typing import Callable, Any, List
from collections import deque


def use_time_check(max_use: int = float("inf")) -> Callable:
    """
    Overview:
        This middleware aims to check the usage times of data in buffer. If the usage times of a data is
        greater than max_use, this data will be removed from buffer as soon as possible.
    """

    def push(next: Callable, data: Any, *args, **kwargs) -> None:
        if 'meta' in kwargs:
            kwargs['meta']['use_count'] = 0
        else:
            kwargs['meta'] = {'use_count': 0}
        return next(data, *args, **kwargs)

    def sample(next: Callable, *args, **kwargs) -> List[Any]:
        kwargs['return_index'] = True
        kwargs['return_meta'] = True
        data = next(*args, **kwargs)
        for i, (d, idx, meta) in enumerate(data):
            meta['use_count'] += 1
            if meta['use_count'] >= max_use:
                print('max_use trigger')  # TODO(nyz)
        return data

    def _immutable_object(action: str, next: Callable, *args, **kwargs) -> Any:
        if action == "push":
            return push(next, *args, **kwargs)
        elif action == "sample":
            return sample(next, *args, **kwargs)
        return next(*args, **kwargs)

    return _immutable_object

from typing import Callable, Any, List
from collections import deque


def use_time_check(max_use: int = float("inf")) -> Callable:
    """
    Overview:
        This middleware aims to check the usage times of data in buffer. If the usage times of a data is
        greater than max_use, this data will be removed from buffer as soon as possible.
    """

    def push(chain: Callable, data: Any, *args, **kwargs) -> None:
        if 'meta' in kwargs:
            kwargs['meta']['use_count'] = 0
        else:
            kwargs['meta'] = {'use_count': 0}
        return chain(data, *args, **kwargs)

    def sample(chain: Callable, *args, **kwargs) -> List[Any]:
        kwargs['return_index'] = True
        kwargs['return_meta'] = True
        data = chain(*args, **kwargs)
        for i, (d, idx, meta) in enumerate(data):
            meta['use_count'] += 1
            if meta['use_count'] >= max_use:
                print('max_use trigger')  # TODO(nyz)
        return data

    def _immutable_object(action: str, chain: Callable, *args, **kwargs) -> Any:
        if action == "push":
            return push(chain, *args, **kwargs)
        elif action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _immutable_object

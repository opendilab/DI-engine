from typing import Callable, Any, List


def use_time_check(buffer_: 'Buffer', max_use: int = float("inf")) -> Callable:  # noqa
    """
    Overview:
        This middleware aims to check the usage times of data in buffer. If the usage times of a data is
        greater than or equal to max_use, this data will be removed from buffer as soon as possible.
    """

    def push(chain: Callable, data: Any, meta: dict = None, *args, **kwargs) -> Any:
        if meta:
            meta["use_count"] = 0
        else:
            meta = {"use_count": 0}
        return chain(data, meta, *args, **kwargs)

    def sample(chain: Callable, *args, **kwargs) -> List[Any]:
        data = chain(*args, **kwargs)

        for item in data:
            meta, idx = item.meta, item.index
            meta['use_count'] += 1
            if meta['use_count'] >= max_use:
                buffer_.delete(idx)
        return data

    def _use_time_check(action: str, chain: Callable, *args, **kwargs) -> Any:
        if action == "push":
            return push(chain, *args, **kwargs)
        elif action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _use_time_check

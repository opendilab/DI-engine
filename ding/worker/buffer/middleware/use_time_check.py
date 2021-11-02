from typing import Callable, Any, List


def use_time_check(buffer_: 'Buffer', max_use: int = float("inf")) -> Callable:  # noqa
    """
    Overview:
        This middleware aims to check the usage times of data in buffer. If the usage times of a data is
        greater than or equal to max_use, this data will be removed from buffer as soon as possible.
    """

    def push(chain: Callable, data: Any, *args, **kwargs) -> None:
        if 'meta' in kwargs:
            kwargs['meta']['use_count'] = 0
        else:
            kwargs['meta'] = {'use_count': 0}
        return chain(data, *args, **kwargs)

    def sample(chain: Callable, *args, **kwargs) -> List[Any]:
        data = chain(return_index=True, return_meta=True, *args, **kwargs)

        for i, (d, idx, meta) in enumerate(data):
            meta['use_count'] += 1
            if meta['use_count'] >= max_use:
                buffer_.delete(idx)

        return_index = kwargs.get('return_index', False)
        return_meta = kwargs.get('return_meta', False)
        if return_index and not return_meta:
            data = list(map(lambda item: (item[0], item[1]), data))
        elif not return_index and return_meta:
            data = list(map(lambda item: (item[0], item[2]), data))
        elif not return_index and not return_meta:
            data = list(map(lambda item: item[0], data))

        return data

    def _use_time_check(action: str, chain: Callable, *args, **kwargs) -> Any:
        if action == "push":
            return push(chain, *args, **kwargs)
        elif action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _use_time_check

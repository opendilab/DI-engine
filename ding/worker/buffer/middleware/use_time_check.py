from collections import defaultdict
from typing import Callable, Any, List, Union
from ding.worker.buffer import BufferedData


def use_time_check(buffer_: 'Buffer', max_use: int = float("inf")) -> Callable:  # noqa
    """
    Overview:
        This middleware aims to check the usage times of data in buffer. If the usage times of a data is
        greater than or equal to max_use, this data will be removed from buffer as soon as possible.
    """
    use_count = defaultdict(int)

    def _check_use_count(item: BufferedData):
        nonlocal use_count
        idx = item.index
        use_count[idx] += 1
        item.meta['use_count'] = use_count[idx]
        if use_count[idx] >= max_use:
            buffer_.delete(idx)
            del use_count[idx]

    def sample(chain: Callable, *args, **kwargs) -> Union[List[BufferedData], List[List[BufferedData]]]:
        sampled_data = chain(*args, **kwargs)
        if len(sampled_data) == 0:
            return sampled_data

        if isinstance(sampled_data[0], BufferedData):
            for item in sampled_data:
                _check_use_count(item)
        else:
            for grouped_data in sampled_data:
                for item in grouped_data:
                    _check_use_count(item)
        return sampled_data

    def _use_time_check(action: str, chain: Callable, *args, **kwargs) -> Any:
        if action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _use_time_check

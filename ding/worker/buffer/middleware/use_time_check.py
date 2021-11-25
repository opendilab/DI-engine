from collections import defaultdict
from typing import Callable, Any, List, Optional, Union
from ding.worker.buffer import BufferedData


def use_time_check(buffer_: 'Buffer', max_use: int = float("inf")) -> Callable:  # noqa
    """
    Overview:
        This middleware aims to check the usage times of data in buffer. If the usage times of a data is
        greater than or equal to max_use, this data will be removed from buffer as soon as possible.
    """
    use_count = defaultdict(int)

    def _need_delete(item: BufferedData) -> bool:
        nonlocal use_count
        idx = item.index
        use_count[idx] += 1
        item.meta['use_count'] = use_count[idx]
        if use_count[idx] >= max_use:
            return True
        else:
            return False

    def _check_use_count(sampled_data: List[BufferedData]):
        delete_indices = [item.index for item in filter(_need_delete, sampled_data)]
        buffer_.delete(delete_indices)
        for index in delete_indices:
            del use_count[index]

    def sample(chain: Callable, *args, **kwargs) -> Union[List[BufferedData], List[List[BufferedData]]]:
        sampled_data = chain(*args, **kwargs)
        if len(sampled_data) == 0:
            return sampled_data

        if isinstance(sampled_data[0], BufferedData):
            _check_use_count(sampled_data)
        else:
            for grouped_data in sampled_data:
                _check_use_count(grouped_data)
        return sampled_data

    def _use_time_check(action: str, chain: Callable, *args, **kwargs) -> Any:
        if action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _use_time_check

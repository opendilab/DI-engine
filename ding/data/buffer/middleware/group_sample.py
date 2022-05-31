import random
from typing import Callable, List
from ding.data.buffer.buffer import BufferedData


def group_sample(size_in_group: int, ordered_in_group: bool = True, max_use_in_group: bool = True) -> Callable:
    """
    Overview:
        The middleware is designed to process the data in each group after sampling from the buffer.
    Arguments:
        - size_in_group (:obj:`int`): Sample size in each group.
        - ordered_in_group (:obj:`bool`): Whether to keep the original order of records, default is true.
        - max_use_in_group (:obj:`bool`): Whether to use as much data in each group as possible, default is true.
    """

    def sample(chain: Callable, *args, **kwargs) -> List[List[BufferedData]]:
        if not kwargs.get("groupby"):
            raise Exception("Group sample must be used when the `groupby` parameter is specified.")
        sampled_data = chain(*args, **kwargs)
        for i, grouped_data in enumerate(sampled_data):
            if ordered_in_group:
                if max_use_in_group:
                    end = max(0, len(grouped_data) - size_in_group) + 1
                else:
                    end = len(grouped_data)
                start_idx = random.choice(range(end))
                sampled_data[i] = grouped_data[start_idx:start_idx + size_in_group]
            else:
                sampled_data[i] = random.sample(grouped_data, k=size_in_group)
        return sampled_data

    def _group_sample(action: str, chain: Callable, *args, **kwargs):
        if action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _group_sample

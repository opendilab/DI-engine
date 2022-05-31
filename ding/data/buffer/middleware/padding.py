import random
from typing import Callable, Union, List

from ding.data.buffer import BufferedData
from ding.utils import fastcopy


def padding(policy="random"):
    """
    Overview:
        Fill the nested buffer list to the same size as the largest list.
        The default policy `random` will randomly select data from each group
        and fill it into the current group list.
    Arguments:
        - policy (:obj:`str`): Padding policy, supports `random`, `none`.
    """

    def sample(chain: Callable, *args, **kwargs) -> Union[List[BufferedData], List[List[BufferedData]]]:
        sampled_data = chain(*args, **kwargs)
        if len(sampled_data) == 0 or isinstance(sampled_data[0], BufferedData):
            return sampled_data
        max_len = len(max(sampled_data, key=len))
        for i, grouped_data in enumerate(sampled_data):
            group_len = len(grouped_data)
            if group_len == max_len:
                continue
            for _ in range(max_len - group_len):
                if policy == "random":
                    sampled_data[i].append(fastcopy.copy(random.choice(grouped_data)))
                elif policy == "none":
                    sampled_data[i].append(BufferedData(data=None, index=None, meta=None))

        return sampled_data

    def _padding(action: str, chain: Callable, *args, **kwargs):
        if action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _padding

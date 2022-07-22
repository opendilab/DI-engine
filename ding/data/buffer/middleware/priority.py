from typing import Callable, Any, List, Dict, Optional, Union, TYPE_CHECKING
import copy
import numpy as np
import torch
from ding.utils import SumSegmentTree, MinSegmentTree
from ding.data.buffer.buffer import BufferedData
if TYPE_CHECKING:
    from ding.data.buffer.buffer import Buffer


class PriorityExperienceReplay:
    """
    Overview:
        The middleware that implements priority experience replay (PER).
    """

    def __init__(
            self,
            buffer: 'Buffer',
            IS_weight: bool = True,
            priority_power_factor: float = 0.6,
            IS_weight_power_factor: float = 0.4,
            IS_weight_anneal_train_iter: int = int(1e5),
    ) -> None:
        """
        Arguments:
            - buffer (:obj:`Buffer`): The buffer to use PER.
            - IS_weight (:obj:`bool`): Whether use importance sampling or not.
            - priority_power_factor (:obj:`float`): The factor that adjust the sensitivity between\
                the sampling probability and the priority level.
            - IS_weight_power_factor (:obj:`float`): The factor that adjust the sensitivity between\
                the sample rarity and sampling probability in importance sampling.
            - IS_weight_anneal_train_iter (:obj:`float`): The factor that controls the increasing of\
                ``IS_weight_power_factor`` during training.
        """

        self.buffer = buffer
        self.buffer_idx = {}
        self.buffer_size = buffer.size
        self.IS_weight = IS_weight
        self.priority_power_factor = priority_power_factor
        self.IS_weight_power_factor = IS_weight_power_factor
        self.IS_weight_anneal_train_iter = IS_weight_anneal_train_iter

        # Max priority till now, it's used to initizalize data's priority if "priority" is not passed in with the data.
        self.max_priority = 1.0
        # Capacity needs to be the power of 2.
        capacity = int(np.power(2, np.ceil(np.log2(self.buffer_size))))
        self.sum_tree = SumSegmentTree(capacity)
        if self.IS_weight:
            self.min_tree = MinSegmentTree(capacity)
            self.delta_anneal = (1 - self.IS_weight_power_factor) / self.IS_weight_anneal_train_iter
        self.pivot = 0

    def push(self, chain: Callable, data: Any, meta: Optional[dict] = None, *args, **kwargs) -> BufferedData:
        if meta is None:
            meta = {'priority': self.max_priority}
        else:
            if 'priority' not in meta:
                meta['priority'] = self.max_priority
        meta['priority_idx'] = self.pivot
        self._update_tree(meta['priority'], self.pivot)
        buffered = chain(data, meta=meta, *args, **kwargs)
        index = buffered.index
        self.buffer_idx[self.pivot] = index
        self.pivot = (self.pivot + 1) % self.buffer_size
        return buffered

    def sample(self, chain: Callable, size: int, *args,
               **kwargs) -> Union[List[BufferedData], List[List[BufferedData]]]:
        # Divide [0, 1) into size intervals on average
        intervals = np.array([i * 1.0 / size for i in range(size)])
        # Uniformly sample within each interval
        mass = intervals + np.random.uniform(size=(size, )) * 1. / size
        # Rescale to [0, S), where S is the sum of all datas' priority (root value of sum tree)
        mass *= self.sum_tree.reduce()
        indices = [self.sum_tree.find_prefixsum_idx(m) for m in mass]
        indices = [self.buffer_idx[i] for i in indices]
        # Sample with indices
        data = chain(indices=indices, *args, **kwargs)
        if self.IS_weight:
            # Calculate max weight for normalizing IS
            sum_tree_root = self.sum_tree.reduce()
            p_min = self.min_tree.reduce() / sum_tree_root
            buffer_count = self.buffer.count()
            max_weight = (buffer_count * p_min) ** (-self.IS_weight_power_factor)
            for i in range(len(data)):
                meta = data[i].meta
                priority_idx = meta['priority_idx']
                p_sample = self.sum_tree[priority_idx] / sum_tree_root
                weight = (buffer_count * p_sample) ** (-self.IS_weight_power_factor)
                meta['priority_IS'] = weight / max_weight
                data[i].data['priority_IS'] = torch.as_tensor([meta['priority_IS']]).float()  # for compability
            self.IS_weight_power_factor = min(1.0, self.IS_weight_power_factor + self.delta_anneal)
        return data

    def update(self, chain: Callable, index: str, data: Any, meta: Any, *args, **kwargs) -> None:
        update_flag = chain(index, data, meta, *args, **kwargs)
        if update_flag:  # when update succeed
            assert meta is not None, "Please indicate dict-type meta in priority update"
            new_priority, idx = meta['priority'], meta['priority_idx']
            assert new_priority >= 0, "new_priority should greater than 0, but found {}".format(new_priority)
            new_priority += 1e-5  # Add epsilon to avoid priority == 0
            self._update_tree(new_priority, idx)
            self.max_priority = max(self.max_priority, new_priority)

    def delete(self, chain: Callable, index: str, *args, **kwargs) -> None:
        for item in self.buffer.storage:
            meta = item.meta
            priority_idx = meta['priority_idx']
            self.sum_tree[priority_idx] = self.sum_tree.neutral_element
            self.min_tree[priority_idx] = self.min_tree.neutral_element
            self.buffer_idx.pop(priority_idx)
        return chain(index, *args, **kwargs)

    def clear(self, chain: Callable) -> None:
        self.max_priority = 1.0
        capacity = int(np.power(2, np.ceil(np.log2(self.buffer_size))))
        self.sum_tree = SumSegmentTree(capacity)
        if self.IS_weight:
            self.min_tree = MinSegmentTree(capacity)
        self.buffer_idx = {}
        self.pivot = 0
        chain()

    def _update_tree(self, priority: float, idx: int) -> None:
        weight = priority ** self.priority_power_factor
        self.sum_tree[idx] = weight
        if self.IS_weight:
            self.min_tree[idx] = weight

    def state_dict(self) -> Dict:
        return {
            'max_priority': self.max_priority,
            'IS_weight_power_factor': self.IS_weight_power_factor,
            'sumtree': self.sumtree,
            'mintree': self.mintree,
            'buffer_idx': self.buffer_idx,
        }

    def load_state_dict(self, _state_dict: Dict, deepcopy: bool = False) -> None:
        for k, v in _state_dict.items():
            if deepcopy:
                setattr(self, '{}'.format(k), copy.deepcopy(v))
            else:
                setattr(self, '{}'.format(k), v)

    def __call__(self, action: str, chain: Callable, *args, **kwargs) -> Any:
        if action in ["push", "sample", "update", "delete", "clear"]:
            return getattr(self, action)(chain, *args, **kwargs)
        return chain(*args, **kwargs)

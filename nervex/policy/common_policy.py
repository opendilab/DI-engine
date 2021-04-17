from typing import List, Dict, Any, Tuple, Union, Optional
import copy
from collections import deque

from nervex.data import default_collate, default_decollate
from nervex.torch_utils import to_device
from easydict import EasyDict
from .base_policy import Policy


class CommonPolicy(Policy):

    def _data_preprocess_learn(self, data: List[Any]) -> Tuple[dict, dict]:
        data_info = {
            'replay_buffer_idx': [d.get('replay_buffer_idx', None) for d in data],
            'replay_unique_id': [d.get('replay_unique_id', None) for d in data],
        }
        # data preprocess
        data = default_collate(data)
        ignore_done = self._cfg.learn.get('ignore_done', False)
        if ignore_done:
            data['done'] = None
        else:
            data['done'] = data['done'].float()
        use_priority = self._cfg.get('use_priority', False)
        if use_priority:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)
        if self._use_cuda:
            data = to_device(data, self._device)
        return data, data_info

    def _data_preprocess_collect(self, data: Dict[int, Any]) -> Tuple[List[int], dict]:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        return data_id, data

    def _data_postprocess_collect(self, data_id: List[int], data: dict) -> Dict[int, dict]:
        if self._use_cuda:
            data = to_device(data, 'cpu')
        data = default_decollate(data)
        return {i: d for i, d in zip(data_id, data)}

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        # adder is defined in _init_collect
        return self._adder.get_train_sample(data)

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._armor.mode(train=True)
        self._armor.reset(data_id=data_id)

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_armor.mode(train=False)
        self._collect_armor.reset(data_id=data_id)

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_armor.mode(train=False)
        self._eval_armor.reset(data_id=data_id)

from typing import List, Dict, Any, Tuple, Union, Optional
import copy
from collections import deque

from nervex.data import default_collate, default_decollate
from nervex.torch_utils import to_device
from easydict import EasyDict
from .base_policy import Policy


class CommonPolicy(Policy):

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        # data preprocess
        data = default_collate(data)
        if self._use_cuda:
            data = to_device(data, 'cuda')
        ignore_done = self._cfg.learn.get('ignore_done', False)
        if ignore_done:
            data['done'] = None
        else:
            data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        return data

    def _data_preprocess_collect(self, data: Dict[int, Any]) -> Tuple[List[int], dict]:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, 'cuda')
        data = {'obs': data}
        return data_id, data

    def _data_postprocess_collect(self, data_id: List[int], data: dict) -> Dict[int, dict]:
        if self._use_cuda:
            data = to_device(data, 'cpu')
        data = default_decollate(data)
        return {i: d for i, d in zip(data_id, data)}

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        # adder is defined in _init_collect
        data = self._adder.get_traj(traj_cache, self._traj_len)
        return self._adder.get_train_sample(data)

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._agent.mode(train=True)
        self._agent.reset(data_id=data_id)

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_agent.mode(train=False)
        self._collect_agent.reset(data_id=data_id)

    def _get_setting_learn(self, *args, **kwargs) -> dict:
        return {}

    def _get_setting_collect(self, *args, **kwargs) -> dict:
        return {}

    def _get_setting_eval(self, *args, **kwargs) -> dict:
        return {}

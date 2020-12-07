from typing import List, Dict, Any, Tuple, Union
from .base_policy import Policy
from nervex.data import default_collate, default_decollate
from nervex.torch_utils import to_device


class CommonPolicy(Policy):

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        # data preprocess
        data = default_collate(data)
        if self._use_cuda:
            data = to_device(data, 'cuda')
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

    def _get_trajectory(self, transitions: List[dict], done: bool) -> Union[None, List[Any]]:
        if not done and len(transitions) < self._get_traj_length:
            return None
        else:
            return transitions

    def _callback_episode_done_collect(self, data_id: int) -> None:
        self._collect_agent.reset([data_id])
        return {}

    def _get_setting_learn(self) -> dict:
        return {}

    def _get_setting_eval(self) -> dict:
        return {}

from typing import List, Dict, Any, Optional
from collections import deque
import warnings
import copy
import torch
from .gae import gae, gae_data


class Adder(object):
    """
    Overview:
        Adder is a component that handled the different transformations and calculations for transitions
        in Actor Module(data generation and processing), such as GAE, n-step return, transition sampling etc.
    Interface:
        __init__, get_gae
    """

    def __init__(
            self,
            use_cuda: bool,
            unroll_len: int,
            last_fn_type: str = 'last',
            null_transition: Optional[dict] = None
    ) -> None:
        """
        Overview:
            initialization method for a adder instance
        Arguments:
            - use_cuda (:obj:`bool`): whether use cuda in all the operations
            - unroll_len (:obj:`int`): learn training unroll length
        """
        self._use_cuda = use_cuda
        self._unroll_len = unroll_len
        self._last_fn_type = last_fn_type
        assert self._last_fn_type in ['last', 'drop', 'null_padding']
        if self._last_fn_type == 'last':
            self._last_buffer = []
        self._null_transition = null_transition

    def _get_null_transition(self, template: dict):
        if self._null_transition is not None:
            return copy.deepcopy(self._null_transition)
        else:
            return {k: None for k in template.keys()}

    def get_traj(self, data: deque, data_id: int, traj_len: int, return_num: int = 0) -> list:
        num = min(traj_len, len(data))  # traj_len can be inf
        traj = [data.popleft() for _ in range(num)]
        for i in range(return_num):
            data.appendleft(copy.deepcopy(traj[-(i + 1)]))
        return traj

    def get_gae(self, data: List[Dict[str, Any]], last_value: torch.Tensor, gamma: float,
                gae_lambda: float) -> List[Dict[str, Any]]:
        """
        Overview:
            get GAE advantage for stacked transitions(T timestep, 1 batch)
        Arguments:
            - data (:obj:`list`): transitions list, each element is a transition with at least value and\
            reward keys
            - last_value (:obj:`torch.Tensor`): the last value(i.e.: the T+1 timestep)
            - gamma (:obj:`float`): the future discount factor
            - gae_lambda (:obj:`float`): gae lambda parameter
        Returns:
            - data (:obj:`list`): transitions list, whose elements own advantage key-value
        """
        value = torch.stack([d['value'] for d in data] + [last_value])
        reward = torch.stack([d['reward'] for d in data])
        if self._use_cuda:
            value = value.cuda()
            reward = reward.cuda()
        adv = gae(gae_data(value, reward), gamma, gae_lambda)
        if self._use_cuda:
            adv = adv.cpu()
        for i in range(len(data)):
            data[i]['adv'] = adv[i]
        return data

    def get_gae_with_default_last_value(self, data: List[Dict[str, Any]], done: bool, gamma: float,
                                        gae_lambda: float) -> List[Dict[str, Any]]:
        if done:
            last_value = torch.zeros(1)
        else:
            last_value = data[-1]['value']
            data = data[:-1]
        return self.get_gae(data, last_value, gamma, gae_lambda)

    def get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._unroll_len == 1:
            return data
        else:
            raise NotImplementedError

    def get_drqn(self, data: List[Dict[str, Any]], drqn_unroll_length: int):
        #TODO
        ret = []
        for i in range(len(data)):
            ret.append(data[i:i + drqn_unroll_length])
            if i + 1 + drqn_unroll_length == len(data):
                break
        for r in ret:
            assert len(r) == drqn_unroll_length
        return ret

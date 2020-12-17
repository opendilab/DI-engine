from typing import List, Dict, Any, Optional
from collections import deque
import warnings
import copy
import torch
from nervex.utils import list_split, lists_to_dicts
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
            - last_fn_type (:obj:`str`): the method type for dealing with last data in a traj
        """
        self._use_cuda = use_cuda
        self._unroll_len = unroll_len
        self._last_fn_type = last_fn_type
        assert self._last_fn_type in ['last', 'drop', 'null_padding']
        self._null_transition = null_transition

    def _get_null_transition(self, template: dict):
        if self._null_transition is not None:
            return copy.deepcopy(self._null_transition)
        else:
            return copy.deepcopy(template)

    def get_traj(self, data: deque, traj_len: int, return_num: int = 0) -> list:
        num = min(traj_len, len(data))  # traj_len can be inf
        traj = [data.popleft() for _ in range(num)]
        for i in range(min(return_num, len(data))):
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

    def get_nstep_return_data(self, data: List[Dict[str, Any]], nstep: int, traj_len: int) -> List[Dict[str, Any]]:
        if traj_len == float('inf') or len(data) < traj_len:
            # episode done case
            fake_data = {'obs': data[-1]['obs'].clone(), 'reward': torch.zeros(1), 'done': True}
            data += [fake_data for _ in range(nstep)]
        for i in range(len(data) - nstep):
            data[i]['next_obs'] = copy.deepcopy(data[i + nstep]['obs'])
            data[i]['reward'] = torch.cat([data[i + j]['reward'] for j in range(nstep)])
            data[i]['done'] = data[i + nstep - 1]['done']
        return data[:-nstep]

    def get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._unroll_len == 1:
            return data
        else:
            split_data, residual = list_split(data, step=self._unroll_len)

            def null_padding():
                template = copy.deepcopy(residual[0])
                template['done'] = True
                template['reward'] = torch.zeros_like(template['reward'])
                null_data = [self._get_null_transition(template) for _ in range(miss_num)]
                return null_data

            if residual is not None:
                miss_num = self._unroll_len - len(residual)
                if self._last_fn_type == 'drop':
                    pass
                elif self._last_fn_type == 'last':
                    if len(split_data) > 0:
                        last_data = copy.deepcopy(split_data[-1][-miss_num:])
                        split_data.append(last_data + residual)
                    else:
                        null_data = null_padding()
                        split_data.append(residual + null_data)
                elif self._last_fn_type == 'null_padding':
                    null_data = null_padding()
                    split_data.append(residual + null_data)
            if len(split_data) > 0:
                split_data = [lists_to_dicts(d) for d in split_data]
            return split_data

from typing import List, Dict, Any
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

    def __init__(self, use_cuda: bool) -> None:
        """
        Overview:
            initialization method for a adder instance
        Arguments:
            - use_cuda (:obj:`bool`): whether use cuda in all the operations
        """
        self._use_cuda = use_cuda

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

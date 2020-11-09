from typing import List, Dict, Any
import torch
from .gae import gae, gae_data


class Adder(object):

    def __init__(self, use_cuda: bool) -> None:
        self._use_cuda = use_cuda

    def get_gae(self, data: List[Dict[str, Any]], last_value: torch.Tensor, gamma: float,
                gae_lambda: float) -> List[Dict[str, Any]]:
        value = torch.stack([d['value'] for d in data] + [last_value])
        reward = torch.stack([d['reward'] for d in data])
        if self._use_cuda:
            value = value.cuda()
            reward = reward.cuda()
        adv = gae(gae_data(value, reward), gamma, gae_lambda)
        if self._use_cuda:
            adv = adv.cpu()
        adv = adv.squeeze(1)
        for i in range(len(data)):
            data[i]['adv'] = adv[i]
        return data

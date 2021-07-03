from typing import Dict
import torch
import torch.nn as nn

from ding.utils import MODEL_REGISTRY
from .q_learning import DQN


@MODEL_REGISTRY.register('sqn')
class SQN(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(SQN, self).__init__()
        self.q0 = DQN(*args, **kwargs)
        self.q1 = DQN(*args, **kwargs)

    def forward(self, data: torch.Tensor) -> Dict:
        output0 = self.q0(data)
        output1 = self.q1(data)
        return {
            'q_value': [output0['logit'], output1['logit']],
            'logit': output0['logit'],
        }

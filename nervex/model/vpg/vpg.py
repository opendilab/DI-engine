from abc import ABC, abstractmethod
import torch.nn as nn


class IVPG(ABC, nn.Module):
    """
    Overview:
        Basic class of vanilla policy gradient class
    """
    def forward(self, inputs: dict, mode: str, **kwargs) -> dict:
        # Note: policy(action) network and value network can shared some parts of layers
        assert mode in ['compute_action', 'compute_value', 'compute_action_value']
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    @abstractmethod
    def compute_action(self, inputs: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def compute_value(self, inputs: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def compute_action_value(self, inputs: dict) -> dict:
        raise NotImplementedError

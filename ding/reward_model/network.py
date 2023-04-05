from typing import Union, Tuple, List, Dict
from easydict import EasyDict

import torch
import torch.nn as nn

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from ding.torch_utils.data_helper import to_tensor
import numpy as np

class FeatureNetwork(nn.Module):
    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType) -> None:
        super(FeatureNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature = FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.feature = ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feature_output = self.feature(obs)
        return feature_output

class RndNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType) -> None:
        super(RndNetwork, self).__init__()
        self.target = FeatureNetwork(obs_shape, hidden_size_list)
        self.predictor = FeatureNetwork(obs_shape, hidden_size_list)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predict_feature, target_feature
    
class RedNetwork(RndNetwork):
    def __init__(self, obs_shape: int, action_shape: int, hidden_size_list: SequenceType) -> None:
        # RED network does not support high dimension obs
        super().__init__(obs_shape+action_shape, hidden_size_list)

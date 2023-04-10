from typing import Union, Tuple, List, Dict, Optional
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from ding.torch_utils.data_helper import to_tensor
import numpy as np


class RepresentationNetwork(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            hidden_size_list: SequenceType,
            activation: Optional[nn.Module] = nn.ReLU()
    ) -> None:
        super(RepresentationNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature = FCEncoder(obs_shape, hidden_size_list, activation=activation)
        elif len(obs_shape) == 3:
            self.feature = ConvEncoder(obs_shape, hidden_size_list, activation=activation)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own Representation Network".
                format(obs_shape)
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feature_output = self.feature(obs)
        return feature_output


class RndNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType) -> None:
        super(RndNetwork, self).__init__()
        self.target = RepresentationNetwork(obs_shape, hidden_size_list)
        self.predictor = RepresentationNetwork(obs_shape, hidden_size_list)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predict_feature = self.predictor(obs)
            target_feature = self.target(obs)
            reward = F.mse_loss(predict_feature, target_feature.detach())
            reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-8)
        return reward

    def learn(self, obs: torch.Tensor) -> torch.Tensor:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        loss = F.mse_loss(predict_feature, target_feature.detach())
        return loss


class RedNetwork(RndNetwork):

    def __init__(
            self,
            obs_shape: int,
            action_shape: int,
            hidden_size_list: SequenceType,
            sigma: Optional[float] = 0.5
    ) -> None:
        # RED network does not support high dimension obs
        super().__init__(obs_shape + action_shape, hidden_size_list)
        self.sigma = sigma

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predict_feature = self.predictor(obs)
            target_feature = self.target(obs)
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            reward = torch.exp(-self.sigma * reward)
        return reward

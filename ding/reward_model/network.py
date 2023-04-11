from typing import Union, Tuple, List, Dict, Optional
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from ding.torch_utils.data_helper import to_tensor
from .reword_model_utils import concat_state_action_pairs
from functools import partial


class RepresentationNetwork(nn.Module):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        hidden_size_list: SequenceType,
        activation: Optional[nn.Module] = nn.ReLU(),
        kernel_size: Optional[SequenceType] = [8, 4, 3],
        stride: Optional[SequenceType] = [4, 2, 1],
    ) -> None:
        super(RepresentationNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature = FCEncoder(obs_shape, hidden_size_list, activation=activation)
        elif len(obs_shape) == 3:
            self.feature = ConvEncoder(
                obs_shape, hidden_size_list, activation=activation, kernel_size=kernel_size, stride=stride
            )
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
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
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


class GailNetwork(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            hidden_size_list: SequenceType,
            kernel_size: Optional[SequenceType] = [8, 4, 3],
            stride: Optional[SequenceType] = [4, 2, 1],
            activation: Optional[nn.Module] = nn.ReLU(),
            action_shape: Optional[int] = None
    ) -> None:
        super(GailNetwork, self).__init__()
        # Gail will need one more fc layer after RepresentationNetwork, and it will use another activation function
        self.act = nn.Sigmoid()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature = RepresentationNetwork(obs_shape, hidden_size_list, activation)
            self.concat_state_action_pairs = concat_state_action_pairs
            self.fc = nn.Linear(hidden_size_list[0], 1)
            self.image_input = False
        elif len(obs_shape) == 3:
            self.feature = RepresentationNetwork(obs_shape, hidden_size_list, activation, kernel_size, stride)
            self.fc = nn.Linear(64 + self.action_size, 1)
            self.image_input = True
            self.action_size = action_shape
            self.obs_size = obs_shape

    def learn(self, train_data: torch.Tensor, expert_data: torch.Tensor) -> torch.Tensor:
        out_1: torch.Tensor = self.forward(train_data)
        loss_1: torch.Tensor = torch.log(out_1 + 1e-8).mean()
        out_2: torch.Tensor = self.forward(expert_data)
        loss_2: torch.Tensor = torch.log(1 - out_2 + 1e-8).mean()

        loss: torch.Tensor = -(loss_1 + loss_2)

        return loss

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.image_input:
            # input: x = [B, 4 x 84 x 84 + self.action_size], last element is action
            actions = data[:, -self.action_size:]  # [B, self.action_size]
            # get observations
            obs = data[:, :-self.action_size]
            obs = obs.reshape([-1] + self.obs_size)  # [B, 4, 84, 84]
            obs = self.feature(obs)
            data = torch.cat([obs, actions], dim=-1)
        else:
            data = self.feature(data)

        data = self.fc(data)
        reward = self.act(data)
        return reward

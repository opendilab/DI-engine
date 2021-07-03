from typing import Union, Tuple
from collections.abc import Iterable
from easydict import EasyDict

import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as I
from torch.distributions.categorical import Categorical

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from .base_reward_model import BaseRewardModel


def collect_states(iterator):  # get total_states
    res = []
    for item in iterator:
        state = item['obs']
        res.append(state)
    return res


class RndNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType) -> None:
        super(RndNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.target = FCEncoder(obs_shape, hidden_size_list)
            self.predictor = FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.target = ConvEncoder(obs_shape, hidden_size_list)
            self.predictor = ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predict_feature, target_feature


@REWARD_MODEL_REGISTRY.register('rnd')
class RndRewardModel(BaseRewardModel):
    config = dict(
        type='rnd',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        # obs_shape=6,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(RndRewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = RndNetwork(config.obs_shape, config.hidden_size_list)
        self.reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_data = []
        self.opt = optim.Adam(self.reward_model.predictor.parameters(), config.learning_rate)

    def _train(self) -> None:
        train_data: list = random.sample(self.train_data, self.cfg.batch_size)
        train_data: torch.Tensor = torch.stack(train_data).to(self.device)
        predict_feature, target_feature = self.reward_model(train_data)
        loss = F.mse_loss(predict_feature, target_feature.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            self._train()
        self.clear_data()

    def estimate(self, data: list) -> None:
        """
        Rewrite the reward key in each row of the data.
        """
        obs = collect_states(data)
        obs = torch.stack(obs).to(self.device)
        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(obs)
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-8)
            reward = reward.to(data[0]['reward'].device)
            reward = torch.chunk(reward, reward.shape[0], dim=0)
        for item, rew in zip(data, reward):
            if self.intrinsic_reward_type == 'add':
                item['reward'] += rew
            elif self.intrinsic_reward_type == 'new':
                item['intrinsic_reward'] = rew
            elif self.intrinsic_reward_type == 'assign':
                item['reward'] = rew

    def collect_data(self, data: list) -> None:
        self.train_data.extend(collect_states(data))

    def clear_data(self) -> None:
        self.train_data.clear()

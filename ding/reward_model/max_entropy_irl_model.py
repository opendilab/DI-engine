from typing import Union, Tuple
from easydict import EasyDict

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from .base_reward_model import BaseRewardModel

class MaxEntropyNN(nn.Module):
    def __init__(
        self, 
        input_size,
        hidden_size = 128, 
        output_size = 1,
    ):
        super(MaxEntropyNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


@REWARD_MODEL_REGISTRY.register('max_entropy')
class MaxEntropyModel(BaseRewardModel):
    config = dict(
        type='max_entropy',
        learning_rate=1e-3,
        # obs_shape=6,
        batch_size = 64,
        hidden_size = 128,
        update_per_collect=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(MaxEntropyModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = MaxEntropyModel(config.input_size, config.hidden_size)
        self.reward_model.to(self.device)
        self.opt = optim.Adam(self.reward_model.parameters(), lr = config.learning_rate)

    def train(self, expert_demo: torch.Tensor, samp: torch.Tensor):

        cost_demo = self.reward_model(expert_demo)
        cost_samp = self.reward_model(samp)

        prob = []

        loss_IOC = torch.mean(cost_demo) + \
            torch.log(torch.mean(torch.exp(-cost_samp)/(prob+1e-7)))
        # UPDATING THE COST FUNCTION
        self.opt.zero_grad()
        loss_IOC.backward()
        self.opt.step()

    def cal_reward(self, samp):
        cost_samp = self.reward_model(samp)
        return cost_samp

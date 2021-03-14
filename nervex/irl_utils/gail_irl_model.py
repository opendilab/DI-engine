import pickle
import random
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from .base_reward_estimate import BaseRewardModel


def concat_state_action_pairs(iterator):
    # concat state and action
    assert isinstance(iterator, Iterable)
    res = []
    for item in iterator:
        state = item['obs']
        action = item['action']
        s_a = torch.cat([state, action.float()], dim=-1)
        res.append(s_a)
    return res


class RewardModelNetwork(nn.Module):

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int) -> None:
        super(RewardModelNetwork, self).__init__()
        self.l1 = nn.Linear(input_dims, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, output_dims)
        self.a1 = nn.Tanh()
        self.a2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.l1(out)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        return out


class GailRewardModel(BaseRewardModel):

    def __init__(self, config: dict, device) -> None:
        super(GailRewardModel, self).__init__()
        self.config = config
        assert device in ["cpu", "cuda"]
        self.device = device
        self.reward_model = RewardModelNetwork(config['input_dims'], config['hidden_dims'], 1)
        self.reward_model.to(self.device)
        self.expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters())

    def load_expert_data(self) -> None:
        with open(self.config['expert_data_path'], 'rb') as f:
            self.expert_data_loader: list = pickle.load(f)
            print("the data size is:", len(self.expert_data_loader))

    def start(self) -> None:
        self.load_expert_data()
        self.expert_data = concat_state_action_pairs(self.expert_data_loader)

    def _train(self, train_data: torch.Tensor, expert_data: torch.Tensor) -> None:
        # calculate loss, here are some hyper-param
        out_1: torch.Tensor = self.reward_model(train_data)
        loss_1: torch.Tensor = torch.log(out_1 + 1e-5).mean()
        out_2: torch.Tensor = self.reward_model(expert_data)
        loss_2: torch.Tensor = torch.log(1 - out_2 + 1e-5).mean()
        loss: torch.Tensor = loss_1 + loss_2

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.config['train_iterations']):
            sample_expert_data: list = random.sample(self.expert_data, self.config['batch_size'])
            sample_train_data: list = random.sample(self.train_data, self.config['batch_size'])
            sample_expert_data = torch.stack(sample_expert_data).to(self.device)
            sample_train_data = torch.stack(sample_train_data).to(self.device)
            self._train(sample_train_data, sample_expert_data)

    def estimate(self, data: list) -> None:
        """
        Rewrite the reward key in each row of the data.
        """
        res = concat_state_action_pairs(data)
        res = torch.stack(res).to(self.device)
        with torch.no_grad():
            reward = self.reward_model(res).squeeze(-1).cpu()
        reward = torch.chunk(reward, reward.shape[0], dim=0)
        for item, rew in zip(data, reward):
            item['reward'] = rew

    def collect_data(self, data: list) -> None:
        self.train_data = concat_state_action_pairs(data)

    def clear_data(self) -> None:
        self.train_data.clear()

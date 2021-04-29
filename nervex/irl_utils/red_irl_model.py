from typing import Dict, List
import pickle
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim

from nervex.utils import REWARD_MODEL_REGISTRY
from .base_reward_estimate import BaseRewardModel


class SENet(nn.Module):
    """support estimation network"""

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int) -> None:
        super(SENet, self).__init__()
        self.l_1 = nn.Linear(input_dims, hidden_dims)
        self.l_2 = nn.Linear(hidden_dims, output_dims)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.l_1(x)
        out = self.act(out)
        out = self.l_2(out)
        out = self.act(out)
        return out


@REWARD_MODEL_REGISTRY.register('red')
class RedRewardModel(BaseRewardModel):
    """
    Overview:
       The implement of reward model in RED(https://arxiv.org/abs/1905.06750)
    """

    def __init__(self, config: Dict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(RedRewardModel, self).__init__()
        self.config: Dict = config
        self.expert_data: List[tuple] = []
        self.device = device
        self.tb_logger = tb_logger
        self.target_net: SENet = SENet(config['input_dims'], config['hidden_dims'], config['output_dims'])
        self.online_net: SENet = SENet(config['input_dims'], config['hidden_dims'], config['output_dims'])
        self.target_net.to(device)
        self.online_net.to(device)
        self.opt: optim.Adam = optim.Adam(self.online_net.parameters())
        self.train_once_flag = False
        self.warning_flag = False

        self.load_expert_data()

    def load_expert_data(self) -> None:
        with open(self.config['expert_data_path'], 'rb') as f:
            self.expert_data = pickle.load(f)
        sample_size = min(len(self.expert_data), self.config['sample_size'])
        self.expert_data = random.sample(self.expert_data, sample_size)
        print('the expert data size is:', len(self.expert_data))

    def _train(self, batch_data: torch.Tensor) -> float:
        with torch.no_grad():
            target = self.target_net(batch_data)
        hat: torch.Tensor = self.online_net(batch_data)
        loss: torch.Tensor = ((hat - target) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(self) -> None:
        if self.train_once_flag:
            if not self.warning_flag:
                logging.warning('RED model should be trained once, we do not train it anymore')
                self.warning_flag = True
        else:
            for i in range(self.config['train_iteration']):
                sample_batch = random.sample(self.expert_data, self.config['batch_size'])
                states_data = []
                actions_data = []
                for item in sample_batch:
                    states_data.append(item['obs'])
                    actions_data.append(item['action'])
                states_tensor: torch.Tensor = torch.stack(states_data).float()
                actions_tensor: torch.Tensor = torch.stack(actions_data).float()
                states_actions_tensor: torch.Tensor = torch.cat([states_tensor, actions_tensor], dim=1)
                states_actions_tensor = states_actions_tensor.to(self.device)
                loss = self._train(states_actions_tensor)
                self.tb_logger.add_scalar('reward_model/red_loss', loss, i)
            self.train_once_flag = True

    def estimate(self, data: list) -> None:
        states_data = []
        actions_data = []
        for item in data:
            states_data.append(item['obs'])
            actions_data.append(item['action'])
        states_tensor = torch.stack(states_data).float()
        actions_tensor = torch.stack(actions_data).float()
        states_actions_tensor = torch.cat([states_tensor, actions_tensor], dim=1)
        states_actions_tensor = states_actions_tensor.to(self.device)
        with torch.no_grad():
            hat_1 = self.online_net(states_actions_tensor)
            hat_2 = self.target_net(states_actions_tensor)
        c = ((hat_1 - hat_2) ** 2).mean(dim=1)
        r = torch.exp(-self.config['sigma'] * c)
        for item, rew in zip(data, r):
            item['reward'] = rew

    def collect_data(self, data) -> None:
        # if online_net is trained continuously, there should be some implementations in collect_data method
        pass

    def clear_data(self):
        # if online_net is trained continuously, there should be some implementations in clear_data method
        pass

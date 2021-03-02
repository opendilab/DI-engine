from .base_reward_estimate import BaseRewardModel
import torch
import torch.nn as nn
import pickle
import numpy as np
import random
import torch.optim as optim


class DFN(nn.Module):

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int) -> None:
        super(DFN, self).__init__()
        self.l1 = nn.Linear(input_dims, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, output_dims)
        self.a1 = nn.Tanh()
        self.a2 = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        out = x
        out = self.l1(out)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        return out


class GailRewardModel(BaseRewardModel):

    def __init__(self, config: dict) -> None:
        super(GailRewardModel, self).__init__()
        self.config = config
        self.reward_model = DFN(config['input_dims'], config['hidden_dims'], 1)
        self.expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters())

    def load_expert_data(self):
        with open(self.config['expert_data_path'], 'rb') as f:
            self.expert_data_loader: list = pickle.load(f)

    def launch(self):
        self.load_expert_data()
        # make data process
        # concat state and action
        res = []
        for item in self.expert_data_loader:
            state: np.ndarray = item[0]
            action = item[1]
            if isinstance(action, np.ndarray):
                s_a = np.concatenate([state, action])
            else:
                s_list = state.tolist()
                s_list.append(action)
                s_a = np.array(s_list)
            res.append(s_a)
        self.expert_data = res

    def _train(self, train_data, expert_data) -> None:
        # calcute loss
        out_1: torch.Tensor = self.reward_model(train_data)
        loss_1: torch.Tensor = torch.log(out_1).mean()
        out_2: torch.Tensor = self.reward_model(expert_data)
        loss_2: torch.Tensor = torch.log(1 - out_2).mean()
        loss: torch.Tensor = loss_1 + loss_2
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self):
        for _ in range(self.config['train_epoches']):
            sample_expert_data: list = random.sample(self.expert_data, self.config['batch_size'])
            sample_train_data: list = random.sample(self.train_data, self.config['batch_size'])
            # make them to tensor
            sample_expert_tensor = torch.tensor(sample_expert_data, dtype=torch.float32)
            sample_train_tensor = torch.tensor(sample_train_data, dtype=torch.float32)
            self._train(sample_train_tensor, sample_expert_tensor)

    def estimate(self, s, a):
        # s, a 处理的问问题，以及device的问题这些都要后期加吧, 这里还有并行化的问题，到时再讨论吧，把代码改好一点的工作交给别人， 还是我自己来做？？
        s_list: list = s.tolist()
        if isinstance(a, np.ndarray):
            a_list: list = a.tolist()
            s_list.extend(a_list)
            s_a = s_list
        else:
            s_list.append(a)
            s_a = s_list
        s_a_tensor = torch.tensor([s_a], dtype=torch.float32)
        return self.reward_model(s_a_tensor)[0].item()

    def collect_data(self, item):
        # item need to process
        state = item[0]
        action = item[1]
        if isinstance(action, np.ndarray):
            s_a = np.concatenate([state, action])
        else:
            s_list = state.tolist()
            s_list.append(action)
            s_a = np.array(s_list)
        self.train_data.append(s_a)

    def clear_data(self):
        self.train_data.clear()

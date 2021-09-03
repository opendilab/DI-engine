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


def collect_states(iterator):
    states = []
    next_states = []
    actions = []
    for item in iterator:
        state = item['obs']
        next_state = item['next_obs']
        action = item['action']
        states.append(state)
        next_states.append(next_state)
        actions.append(action)
    return states, next_states, actions 

class CuriosityNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType, action_shape:int) -> None:
        super(CuriosityNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature= FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.feature= ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )
        self.device = 'cpu' #torch.device('cuda' if use_cuda else 'cpu')
        self.action_shape = action_shape
        feature_output = hidden_size_list[-1]
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_output * 2, 512),
            nn.ReLU(),
            nn.Linear(512, action_shape)
        )
        self.residual = [nn.Sequential(
            nn.Linear(action_shape + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(action_shape + feature_output, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(action_shape + 512, feature_output),
        )

    def forward(self, state, next_state, action_long):
        #state, next_state, action = inputs
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action_long = torch.LongTensor(action_long)
        #action_onehot = torch.FLoatTensor(len(action), self.action_shape, )
        action_onehot = torch.FloatTensor(
            len(action_long), self.action_shape).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action_long.view(len(action_long), -1), 1)
        action = action_onehot
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action


@REWARD_MODEL_REGISTRY.register('curiosity')
class CuriosityRewardModel(BaseRewardModel):
    config = dict(
        type='curiosity',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        # obs_shape=6,
        action_shape=7,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(CuriosityRewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = CuriosityNetwork(config.obs_shape, config.hidden_size_list, config.action_shape)
        self.reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_data = []
        self.train_states = []
        self.train_next_states = []
        self.train_actions = [] 
        self.opt = optim.Adam(self.reward_model.parameters(), config.learning_rate)

    def _train(self) -> None:
        # train_data: list = random.sample(self.train_data, self.cfg.batch_size)
        # train_data: torch.Tensor = torch.stack(train_data).to(self.device)
        train_data_list = [i for i in range(0, len(self.train_states))]
        train_data_index = random.sample(train_data_list, self.cfg.batch_size)
        data_states = [self.train_states[i] for i in train_data_index]
        data_states = torch.stack(data_states)
        data_next_states = [self.train_next_states[i] for i in train_data_index]
        data_next_states = torch.stack(data_next_states)
        data_actions = [self.train_actions[i] for i in train_data_index]
        data_actions = torch.stack(data_actions)
        real_next_state_feature, pred_next_state_feature, pred_action = self.reward_model(data_states, data_next_states, data_actions)
        ce = nn.CrossEntropyLoss(reduction="mean")
        forward_mse = nn.MSELoss()
        inverse_loss = ce(pred_action, data_actions.squeeze(dim=1).long())
        forward_loss = forward_mse(pred_next_state_feature, real_next_state_feature.detach())
        #loss = F.mse_loss(predict_feature, target_feature.detach())
        loss = inverse_loss + forward_loss 
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            self._train()
        self.clear_data()

    def estimate(self, data: list) -> None:
        states, next_states, actions = collect_states(data)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        with torch.no_grad():
            #predict_feature, target_feature = self.reward_model(states, next_states, actions)
            real_next_state_feature, pred_next_state_feature, pred_action = self.reward_model(states, next_states, actions)
            reward = F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(dim=1)
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
        states, next_states, actions  = collect_states(data)
        self.train_states.extend(states)
        self.train_next_states.extend(next_states)
        self.train_actions.extend(actions)

    def clear_data(self) -> None:
        self.train_data.clear()
        self.train_states.clear()
        self.train_next_states.clear()
        self.train_actions.clear() 

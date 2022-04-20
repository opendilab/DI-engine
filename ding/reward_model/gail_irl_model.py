from typing import List, Dict, Any
import pickle
import random
from collections.abc import Iterable
from easydict import EasyDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel
import torch.nn.functional as F
from functools import partial


def concat_state_action_pairs(iterator):
    """
    Overview:
        Concatenate state and action pairs from input.
    Arguments:
        - iterator (:obj:`Iterable`): Iterables with at least ``obs`` and ``action`` tensor keys.
    Returns:
        - res (:obj:`Torch.tensor`): State and action pairs.
    """
    assert isinstance(iterator, Iterable)
    res = []
    for item in iterator:
        state = item['obs'].flatten()  # to allow 3d obs and actions concatenation
        action = item['action']
        s_a = torch.cat([state, action.float()], dim=-1)
        res.append(s_a)
    return res


def concat_state_action_pairs_one_hot(iterator, action_size: int):
    """
    Overview:
        Concatenate state and action pairs from input. Action values are one-hot encoded
    Arguments:
        - iterator (:obj:`Iterable`): Iterables with at least ``obs`` and ``action`` tensor keys.
    Returns:
        - res (:obj:`Torch.tensor`): State and action pairs.
    """
    assert isinstance(iterator, Iterable)
    res = []
    for item in iterator:
        state = item['obs'].flatten()  # to allow 3d obs and actions concatenation
        action = item['action']
        action = torch.Tensor([int(i == action) for i in range(action_size)])
        s_a = torch.cat([state, action], dim=-1)
        res.append(s_a)
    return res


class RewardModelNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(RewardModelNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.a1 = nn.Tanh()
        self.a2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.l1(out)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        return out


class AtariRewardModelNetwork(nn.Module):

    def __init__(self, input_size: int, action_size: int) -> None:
        super(AtariRewardModelNetwork, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64 + self.action_size, 1)  # here we add 1 to take consideration of the action concat
        self.a = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: x = [B, 4 x 84 x 84 + self.action_size], last element is action
        actions = x[:, -self.action_size:]  # [B, self.action_size]
        # get observations
        x = x[:, :-self.action_size]
        x = x.reshape([-1] + self.input_size)  # [B, 4, 84, 84]
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.reshape(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        x = torch.cat([x, actions], dim=-1)
        x = self.fc2(x)
        r = self.a(x)
        return r


@REWARD_MODEL_REGISTRY.register('gail')
class GailRewardModel(BaseRewardModel):
    """
    Overview:
        The Gail reward model class (https://arxiv.org/abs/1606.03476)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``,  ``state_dict``, ``load_state_dict``, ``learn``
    Config:
           == ====================  ========   =============  ================================= =======================
           ID Symbol                Type       Default Value  Description                       Other(Shape)
           == ====================  ========   =============  ================================= =======================
           1  ``type``              str        gail           | RL policy register name, refer  | this arg is optional,
                                                              | to registry ``POLICY_REGISTRY`` | a placeholder
           2  | ``expert_data_``    str        expert_data.   | Path to the expert dataset      | Should be a '.pkl'
              | ``path``                       .pkl           |                                 | file
           3  | ``update_per_``     int        100            | Number of updates per collect   |
              | ``collect``                                   |                                 |
           4  | ``batch_size``      int        64             | Training batch size             |
           5  | ``input_size``      int                       | Size of the input:              |
              |                                               | obs_dim + act_dim               |
           6  | ``target_new_``     int        64             | Collect steps per iteration     |
              | ``data_count``                                |                                 |
           7  | ``hidden_size``     int        128            | Linear model hidden size        |
           8  | ``collect_count``   int        100000         | Expert dataset size             | One entry is a (s,a)
              |                                               |                                 | tuple
           == ====================  ========   =============  ================================= =======================

       """
    config = dict(
        type='gail',
        learning_rate=1e-3,
        update_per_collect=100,
        batch_size=64,
        input_size=4,
        target_new_data_count=64,
        hidden_size=128,
        collect_count=100000,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`EasyDict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`SummaryWriter`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(GailRewardModel, self).__init__()
        self.cfg = config
        assert device in ["cpu", "cuda"] or "cuda" in device
        self.device = device
        self.tb_logger = tb_logger
        obs_shape = config.input_size
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.reward_model = RewardModelNetwork(config.input_size, config.hidden_size, 1)
            self.concat_state_action_pairs = concat_state_action_pairs
        elif len(obs_shape) == 3:
            action_shape = self.cfg.action_size
            self.reward_model = AtariRewardModelNetwork(config.input_size, action_shape)
            self.concat_state_action_pairs = partial(concat_state_action_pairs_one_hot, action_size=action_shape)
        self.reward_model.to(self.device)
        self.expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters(), config.learning_rate)
        self.train_iter = 0

        self.load_expert_data()

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config.data_path`` attribute in self
        Effects:
            This is a side effect function which updates the expert data attribute \
                (i.e. ``self.expert_data``) with ``fn:concat_state_action_pairs``
        """
        with open(self.cfg.data_path + '/expert_data.pkl', 'rb') as f:
            self.expert_data_loader: list = pickle.load(f)
        self.expert_data = self.concat_state_action_pairs(self.expert_data_loader)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'model': self.reward_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.reward_model.load_state_dict(state_dict['model'])

    def learn(self, train_data: torch.Tensor, expert_data: torch.Tensor) -> float:
        """
        Overview:
            Helper function for ``train`` which calculates loss for train data and expert data.
        Arguments:
            - train_data (:obj:`torch.Tensor`): Data used for training
            - expert_data (:obj:`torch.Tensor`): Expert data
        Returns:
            - Combined loss calculated of reward model from using ``train_data`` and ``expert_data``.
        """
        # calculate loss, here are some hyper-param
        out_1: torch.Tensor = self.reward_model(train_data)
        loss_1: torch.Tensor = torch.log(out_1 + 1e-8).mean()
        out_2: torch.Tensor = self.reward_model(expert_data)
        loss_2: torch.Tensor = torch.log(1 - out_2 + 1e-8).mean()
        # log(x) with 0<x<1 is negative, so to reduce this loss we have to minimize the opposite
        loss: torch.Tensor = -(loss_1 + loss_2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(self) -> None:
        """
        Overview:
            Training the Gail reward model. The training and expert data are randomly sampled with designated\
                 batch size abstracted from the ``batch_size`` attribute in ``self.cfg`` and \
                    correspondingly, the ``expert_data`` as well as ``train_data`` attributes initialized ``self`
        Effects:
            - This is a side effect function which updates the reward model and increment the train iteration count.
        """
        for _ in range(self.cfg.update_per_collect):
            sample_expert_data: list = random.sample(self.expert_data, self.cfg.batch_size)
            sample_train_data: list = random.sample(self.train_data, self.cfg.batch_size)
            sample_expert_data = torch.stack(sample_expert_data).to(self.device)
            sample_train_data = torch.stack(sample_train_data).to(self.device)
            loss = self.learn(sample_train_data, sample_expert_data)
            self.tb_logger.add_scalar('reward_model/gail_loss', loss, self.train_iter)
            self.train_iter += 1

    def estimate(self, data: list) -> List[Dict]:
        """
        Overview:
            Estimate reward by rewriting the reward key in each row of the data.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation, with at least \
                 ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        train_data_augmented = self.reward_deepcopy(data)
        res = self.concat_state_action_pairs(train_data_augmented)
        res = torch.stack(res).to(self.device)
        with torch.no_grad():
            reward = self.reward_model(res).squeeze(-1).cpu()
        reward = torch.chunk(reward, reward.shape[0], dim=0)
        for item, rew in zip(train_data_augmented, reward):
            item['reward'] = -torch.log(rew + 1e-8)

        return train_data_augmented

    def collect_data(self, data: list) -> None:
        """
        Overview:
            Collecting training data formatted by  ``fn:concat_state_action_pairs``.
        Arguments:
            - data (:obj:`Any`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self``
        """
        self.train_data.extend(self.concat_state_action_pairs(data))

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.train_data.clear()

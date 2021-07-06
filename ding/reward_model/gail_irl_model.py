import pickle
import random
from collections.abc import Iterable
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


def concat_state_action_pairs(iterator):
    """
    Overview:
        Concate state and action pairs from input.
    Arguments:
        - iterator (:obj:`Iterable`): Iterables with at least ``obs`` and ``action`` tensor keys.
    Returns:
        - res (:obj:`Torch.tensor`): State and action pairs.
    """
    assert isinstance(iterator, Iterable)
    res = []
    for item in iterator:
        state = item['obs']
        action = item['action']
        s_a = torch.cat([state, action.float()], dim=-1)
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


@REWARD_MODEL_REGISTRY.register('gail')
class GailRewardModel(BaseRewardModel):
    """
    Overview:
        The Gail reward model class (https://arxiv.org/abs/1606.03476)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``,
    """
    config = dict(
        type='gail',
        learning_rate=1e-3,
        # expert_data_path='expert_data.pkl'
        update_per_collect=100,
        batch_size=64,
        # input_size=4,
        target_new_data_count=64,
        hidden_size=128,
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
        self.reward_model = RewardModelNetwork(config.input_size, config.hidden_size, 1)
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
            Getting the expert data from ``config.expert_data_path`` attribute in self
        Effects:
            This is a side effect function which updates the expert data attribute \
                (i.e. ``self.expert_data``) with ``fn:concat_state_action_pairs``
        """
        with open(self.cfg.expert_data_path, 'rb') as f:
            self.expert_data_loader: list = pickle.load(f)
        self.expert_data = concat_state_action_pairs(self.expert_data_loader)

    def _train(self, train_data: torch.Tensor, expert_data: torch.Tensor) -> float:
        """
        Overview:
            Helper function for ``train`` which caclulates loss for train data and expert data.
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
        loss: torch.Tensor = loss_1 + loss_2

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
            loss = self._train(sample_train_data, sample_expert_data)
            self.tb_logger.add_scalar('reward_model/gail_loss', loss, self.train_iter)
            self.train_iter += 1

    def estimate(self, data: list) -> None:
        """
        Overview:
            Estimate reward by rewriting the reward key in each row of the data.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation, with at least \
                 ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        res = concat_state_action_pairs(data)
        res = torch.stack(res).to(self.device)
        with torch.no_grad():
            reward = self.reward_model(res).squeeze(-1).cpu()
        reward = torch.chunk(reward, reward.shape[0], dim=0)
        for item, rew in zip(data, reward):
            item['reward'] = rew

    def collect_data(self, data: list) -> None:
        """
        Overview:
            Collecting training data formatted by  ``fn:concat_state_action_pairs``.
        Arguments:
            - data (:obj:`Any`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self``
        """
        self.train_data.extend(concat_state_action_pairs(data))

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.train_data.clear()

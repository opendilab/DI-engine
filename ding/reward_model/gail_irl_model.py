from typing import List, Dict, Any
import pickle
import random
from collections.abc import Iterable
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel
from .reword_model_utils import concat_state_action_pairs
from .network import GailNetwork
import torch.nn.functional as F
from functools import partial


@REWARD_MODEL_REGISTRY.register('gail')
class GailRewardModel(BaseRewardModel):
    """
    Overview:
        The Gail reward model class (https://arxiv.org/abs/1606.03476)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``,  ``state_dict``, ``load_state_dict``, ``learn``
    Config:
        == ====================  ========   =============  =================================== =======================
        ID Symbol                Type       Default Value  Description                         Other(Shape)
        == ====================  ========   =============  =================================== =======================
        1  ``type``              str        gail           | RL policy register name, refer    | this arg is optional,
                                                           | to registry ``POLICY_REGISTRY``   | a placeholder
        2  | ``expert_data_``    str        expert_data.   | Path to the expert dataset        | Should be a '.pkl'
           | ``path``                       .pkl           |                                   | file
        3  | ``learning_rate``   float      0.001          | The step size of gradient descent |
        4  | ``update_per_``     int        100            | Number of updates per collect     |
           | ``collect``                                   |                                   |
        5  | ``batch_size``      int        64             | Training batch size               |
        6  | ``input_size``      int                       | Size of the input:                |
           |                                               | obs_dim + act_dim                 |
        7  | ``target_new_``     int        64             | Collect steps per iteration       |
           | ``data_count``                                |                                   |
        8  | ``hidden_size``     int        128            | Linear model hidden size          |
        9  | ``collect_count``   int        100000         | Expert dataset size               | One entry is a (s,a)
           |                                               |                                   | tuple
        10 | ``clear_buffer_``   int        1              | clear buffer per fixed iters      | make sure replay
           | ``per_iters``                                                                     | buffer's data count
           |                                                                                   | isn't too few.
           |                                                                                   | (code work in entry)
        == ====================  ========   =============  =================================== =======================
    """
    config = dict(
        # (str) RL policy register name, refer to registry ``POLICY_REGISTRY``.
        type='gail',
        # (float) The step size of gradient descent.
        learning_rate=1e-3,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        update_per_collect=100,
        # (int) How many samples in a training batch.
        batch_size=64,
        # (int) Size of the input: obs_dim + act_dim.
        input_size=4,
        # (int) Collect steps per iteration.
        target_new_data_count=64,
        # (int) Linear model hidden size.
        hidden_size=128,
        # (int) Expert dataset size.
        collect_count=100000,
        # (int) Clear buffer per fixed iters.
        clear_buffer_per_iters=1,
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
            self.reward_model = GailNetwork(obs_shape, [config.hidden_size], nn.Tanh())
            self.concat_state_action_pairs = concat_state_action_pairs
        elif len(obs_shape) == 3:
            action_shape = self.cfg.action_size
            self.reward_model = GailNetwork(
                obs_shape, [16, 16, 16, 16, 64], [7, 5, 3, 3], [3, 2, 2, 1], nn.LeakyReLU(), action_shape
            )
            self.concat_state_action_pairs = partial(concat_state_action_pairs, action_size=action_shape, one_hot_=True)
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
        self.expert_data = torch.unbind(self.expert_data, dim=0)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'model': self.reward_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.reward_model.load_state_dict(state_dict['model'])

    def _train(self) -> float:
        """
        Overview:
            Helper function for ``train`` which caclulates loss for train data and expert data.
        Returns:
            - Combined loss calculated of reward model from using ``states_actions_tensor``.
        """
        # sample train and expert data
        sample_expert_data: list = random.sample(self.expert_data, self.cfg.batch_size)
        sample_train_data: list = random.sample(self.train_data, self.cfg.batch_size)
        sample_expert_data = torch.stack(sample_expert_data).to(self.device)
        sample_train_data = torch.stack(sample_train_data).to(self.device)

        loss = self.reward_model.learn(sample_train_data, sample_expert_data)
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
            loss = self._train()
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
        res = res.to(self.device)
        with torch.no_grad():
            reward = self.reward_model.forward(res).squeeze(-1).cpu()
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
        data = self.concat_state_action_pairs(data)
        data = torch.unbind(data, dim=0)
        self.train_data.extend(data)

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.train_data.clear()

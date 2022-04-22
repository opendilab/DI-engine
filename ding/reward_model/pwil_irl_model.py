from typing import Dict, List
import math
import random
import pickle
import torch

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


def collect_state_action_pairs(iterator):
    # concat state and action
    """
    Overview:
        Concate state and action pairs from input iterator.
    Arguments:
        - iterator (:obj:`Iterable`): Iterables with at least ``obs`` and ``action`` tensor keys.
    Returns:
        - res (:obj:`Torch.tensor`): State and action pairs.
    """
    res = []
    for item in iterator:
        state = item['obs']
        action = item['action']
        # s_a = torch.cat([state, action.float()], dim=-1)
        res.append((state, action))
    return res


@REWARD_MODEL_REGISTRY.register('pwil')
class PwilRewardModel(BaseRewardModel):
    """
    Overview:
        The Pwil reward model class (https://arxiv.org/pdf/2006.04678.pdf)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``, ``_get_state_distance``, ``_get_action_distance``
    Properties:
        - reward_table (:obj: `Dict`): In this algorithm, reward model is a dictionary.
    """
    config = dict(
        type='pwil',
        # expert_data_path='expert_data.pkl',
        sample_size=1000,
        alpha=5,
        beta=5,
        # s_size=4,
        # a_size=2,
    )

    def __init__(self, config: Dict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`Dict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`str`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(PwilRewardModel, self).__init__()
        self.cfg: Dict = config
        assert device in ["cpu", "cuda"] or "cuda" in device
        self.device = device
        self.expert_data: List[tuple] = []
        self.train_data: List[tuple] = []
        # In this algo, model is a dict
        self.reward_table: Dict = {}
        self.T: int = 0

        self.load_expert_data()

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config['expert_data_path']`` attribute in self
        Effects:
            This is a side effect function which updates the expert data attribute (e.g.  ``self.expert_data``); \
            in this algorithm, also the ``self.expert_s``, ``self.expert_a`` for states and actions are updated.

        """
        with open(self.cfg.expert_data_path, 'rb') as f:
            self.expert_data = pickle.load(f)
            print("the data size is:", len(self.expert_data))
        sample_size = min(self.cfg.sample_size, len(self.expert_data))
        self.expert_data = random.sample(self.expert_data, sample_size)
        self.expert_data = [(item['obs'], item['action']) for item in self.expert_data]
        self.expert_s, self.expert_a = list(zip(*self.expert_data))
        print('the expert data demonstrations is:', len(self.expert_data))

    def collect_data(self, data: list) -> None:
        """
        Overview:
            Collecting training data formatted by  ``fn:concat_state_action_pairs``.
        Arguments:
            - data (:obj:`list`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self``; \
                in this algorithm, also the ``s_size``, ``a_size`` for states and actions are updated in the \
                    attribute in ``self.cfg`` Dict; ``reward_factor`` also updated as ``collect_data`` called.
        """
        self.train_data.extend(collect_state_action_pairs(data))
        self.T = len(self.train_data)

        s_size = self.cfg.s_size
        a_size = self.cfg.a_size
        beta = self.cfg.beta
        self.reward_factor = -beta * self.T / math.sqrt(s_size + a_size)

    def train(self) -> None:
        """
        Overview:
            Training the Pwil reward model.
        """
        self._train(self.train_data)

    def estimate(self, data: list) -> List[Dict]:
        """
        Overview:
            Estimate reward by rewriting the reward key in each row of the data.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation, \
                with at least ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the ``reward_table`` with ``(obs,action)`` \
                tuples from input.
        """
        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        train_data_augmented = self.reward_deepcopy(data)
        for item in train_data_augmented:
            s = item['obs']
            a = item['action']
            if (s, a) in self.reward_table:
                item['reward'] = self.reward_table[(s, a)]
            else:
                # when (s, a) pair is not trained, set the reward value to default value(e.g.: 0)
                item['reward'] = torch.zeros_like(item['reward'])
        return train_data_augmented

    def _get_state_distance(self, s1: list, s2: list) -> torch.Tensor:
        """
        Overview:
            Getting distances of states given 2 state lists. One single state \
                is of shape ``torch.Size([n])`` (``n`` referred in in-code comments)
        Arguments:
            - s1 (:obj:`torch.Tensor list`): the 1st states' list of size M
            - s2 (:obj:`torch.Tensor list`): the 2nd states' list of size N
        Returns:
            - distance (:obj:`torch.Tensor`) Euclidean distance tensor of  \
                the state tensor lists, of size M x N.
        """
        # Format the values in the tensors to be of float type
        s1 = torch.stack(s1).float()
        s2 = torch.stack(s2).float()
        M, N = s1.shape[0], s2.shape[0]
        # Automatically fill in length
        s1 = s1.view(M, -1)
        s2 = s2.view(N, -1)
        # Automatically fill in & format the tensor size to be (MxNxn)
        s1 = s1.unsqueeze(1).repeat(1, N, 1)
        s2 = s2.unsqueeze(0).repeat(M, 1, 1)
        # Return the distance tensor of size MxN
        return ((s1 - s2) ** 2).mean(dim=-1)

    def _get_action_distance(self, a1: list, a2: list) -> torch.Tensor:
        # TODO the metric of action distance maybe different from envs
        """
        Overview:
            Getting distances of actions given 2 action lists. One single action \
                is of shape ``torch.Size([n])`` (``n`` referred in in-code comments)
        Arguments:
            - a1 (:obj:`torch.Tensor list`): the 1st actions' list of size M
            - a2 (:obj:`torch.Tensor list`): the 2nd actions' list of size N
        Returns:
            - distance (:obj:`torch.Tensor`) Euclidean distance tensor of  \
                the action tensor lists, of size M x N.
        """
        a1 = torch.stack(a1).float()
        a2 = torch.stack(a2).float()
        M, N = a1.shape[0], a2.shape[0]
        a1 = a1.view(M, -1)
        a2 = a2.view(N, -1)
        a1 = a1.unsqueeze(1).repeat(1, N, 1)
        a2 = a2.unsqueeze(0).repeat(M, 1, 1)
        return ((a1 - a2) ** 2).mean(dim=-1)

    def _train(self, data: list):
        """
        Overview:
            Helper function for ``train``, find the min disctance ``s_e``, ``a_e``.
        Arguments:
            - data (:obj:`list`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the ``reward_table`` attribute in ``self`` .
        """
        batch_s, batch_a = list(zip(*data))
        s_distance_matrix = self._get_state_distance(batch_s, self.expert_s)
        a_distance_matrix = self._get_action_distance(batch_a, self.expert_a)
        distance_matrix = s_distance_matrix + a_distance_matrix
        w_e_list = [1 / len(self.expert_data)] * len(self.expert_data)
        for i, item in enumerate(data):
            s, a = item
            w_pi = 1 / self.T
            c = 0
            expert_data_idx = torch.arange(len(self.expert_data)).tolist()
            while w_pi > 0:
                selected_dist = distance_matrix[i, expert_data_idx]
                nearest_distance = selected_dist.min().item()
                nearest_index_selected = selected_dist.argmin().item()
                nearest_index = expert_data_idx[nearest_index_selected]
                if w_pi >= w_e_list[nearest_index]:
                    c = c + nearest_distance * w_e_list[nearest_index]
                    w_pi = w_pi - w_e_list[nearest_index]
                    expert_data_idx.pop(nearest_index_selected)
                else:
                    c = c + w_pi * nearest_distance
                    w_e_list[nearest_index] = w_e_list[nearest_index] - w_pi
                    w_pi = 0
            reward = self.cfg.alpha * math.exp(self.reward_factor * c)
            self.reward_table[(s, a)] = torch.FloatTensor([reward])

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.train_data.clear()

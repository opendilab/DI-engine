from collections.abc import Iterable
from easydict import EasyDict
import numpy as np
import pickle
from copy import deepcopy
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent
from torch.distributions.categorical import Categorical

from ding.utils import REWARD_MODEL_REGISTRY
from ding.model.template.q_learning import DQN
from ding.model.template.vac import VAC
from ding.model.template.qac import QAC
from ding.utils import SequenceType
from ding.model.common import FCEncoder
from ding.utils.data import offline_data_save_type
from ding.utils import build_logger
from ding.utils.data import default_collate

from .base_reward_model import BaseRewardModel
from .rnd_reward_model import collect_states


class TrexConvEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [16, 16, 16, 16, 64, 1],
            activation: Optional[nn.Module] = nn.LeakyReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the Trex Convolution Encoder according to arguments. TrexConvEncoder is different \
                from the ConvEncoder in model.common.encoder, their stride and kernel size parameters \
                are different
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, some ``output size``
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - activation (:obj:`nn.Module`):
                The type of activation to use in the conv ``layers``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResBlock`` for more details
        """
        super(TrexConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list

        layers = []
        kernel_size = [7, 5, 3, 3]
        stride = [3, 2, 1, 1]
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i]))
            layers.append(self.act)
            input_size = hidden_size_list[i]
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.mid = nn.Sequential(
            nn.Linear(flatten_size, hidden_size_list[-2]), self.act,
            nn.Linear(hidden_size_list[-2], hidden_size_list[-1])
        )

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Arguments:
            - x (:obj:`torch.Tensor`): Encoded Tensor after ``self.main``
        Returns:
            - outputs (:obj:`torch.Tensor`): Size int, also number of in-feature
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.main(x)
        x = self.mid(x)
        return x


class TrexModel(nn.Module):

    def __init__(self, obs_shape):
        super(TrexModel, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = nn.Sequential(FCEncoder(obs_shape, [512, 64]), nn.Linear(64, 1))
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = TrexConvEncoder(obs_shape)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own Trex model".
                format(obs_shape)
            )

    def cum_return(self, traj: torch.Tensor, mode: str = 'sum') -> Tuple[torch.Tensor, torch.Tensor]:
        '''calculate cumulative return of trajectory'''
        r = self.encoder(traj)
        if mode == 'sum':
            sum_rewards = torch.sum(r)
            sum_abs_rewards = torch.sum(torch.abs(r))
            return sum_rewards, sum_abs_rewards
        elif mode == 'batch':
            return r, torch.abs(r)
        else:
            raise KeyError("not support mode: {}, please choose mode=sum or mode=batch".format(mode))

    def forward(self, traj_i: torch.Tensor, traj_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j


@REWARD_MODEL_REGISTRY.register('trex')
class TrexRewardModel(BaseRewardModel):
    """
    Overview:
        The Trex reward model class (https://arxiv.org/pdf/1904.06387.pdf)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``,
    """
    config = dict(
        type='trex',
        learning_rate=1e-5,
        update_per_collect=100,
        num_trajs=0,  # number of downsampled full trajectories
        num_snippets=6000,  # number of short subtrajectories to sample
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
        super(TrexRewardModel, self).__init__()
        self.cfg = config
        assert device in ["cpu", "cuda"] or "cuda" in device
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = TrexModel(self.cfg.policy.model.obs_shape)
        self.reward_model.to(self.device)
        self.pre_expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters(), config.reward_model.learning_rate)
        self.train_iter = 0
        self.learning_returns = []
        self.training_obs = []
        self.training_labels = []
        self.num_trajs = self.cfg.reward_model.num_trajs
        self.num_snippets = self.cfg.reward_model.num_snippets
        # minimum number of short subtrajectories to sample
        self.min_snippet_length = config.reward_model.min_snippet_length
        # maximum number of short subtrajectories to sample
        self.max_snippet_length = config.reward_model.max_snippet_length
        self.l1_reg = 0
        self.data_for_save = {}
        self._logger, self._tb_logger = build_logger(
            path='./{}/log/{}'.format(self.cfg.exp_name, 'trex_reward_model'), name='trex_reward_model'
        )
        self.load_expert_data()

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config.data_path`` attribute in self
        Effects:
            This is a side effect function which updates the expert data attribute \
                (i.e. ``self.expert_data``) with ``fn:concat_state_action_pairs``
        """
        with open(self.cfg.reward_model.data_path + '/episodes_data.pkl', 'rb') as f:
            self.pre_expert_data = pickle.load(f)
        with open(self.cfg.reward_model.data_path + '/learning_returns.pkl', 'rb') as f:
            self.learning_returns = pickle.load(f)

        self.create_training_data()
        self._logger.info("num_training_obs: {}".format(len(self.training_obs)))
        self._logger.info("num_labels: {}".format(len(self.training_labels)))

    def create_training_data(self):
        num_trajs = self.num_trajs
        num_snippets = self.num_snippets
        min_snippet_length = self.min_snippet_length
        max_snippet_length = self.max_snippet_length

        demo_lengths = []
        for i in range(len(self.pre_expert_data)):
            demo_lengths.append([len(d) for d in self.pre_expert_data[i]])

        self._logger.info("demo_lengths: {}".format(demo_lengths))
        max_snippet_length = min(np.min(demo_lengths), max_snippet_length)
        self._logger.info("min snippet length: {}".format(min_snippet_length))
        self._logger.info("max snippet length: {}".format(max_snippet_length))

        # collect training data
        max_traj_length = 0
        num_bins = len(self.pre_expert_data)
        assert num_bins >= 2

        # add full trajs (for use on Enduro)
        si = np.random.randint(6, size=num_trajs)
        sj = np.random.randint(6, size=num_trajs)
        step = np.random.randint(3, 7, size=num_trajs)
        for n in range(num_trajs):
            # pick two random demonstrations
            bi, bj = np.random.choice(num_bins, size=(2, ), replace=False)
            ti = np.random.choice(len(self.pre_expert_data[bi]))
            tj = np.random.choice(len(self.pre_expert_data[bj]))
            # create random partial trajs by finding random start frame and random skip frame
            traj_i = self.pre_expert_data[bi][ti][si[n]::step[n]]  # slice(start,stop,step)
            traj_j = self.pre_expert_data[bj][tj][sj[n]::step[n]]

            label = int(bi <= bj)

            self.training_obs.append((traj_i, traj_j))
            self.training_labels.append(label)
            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

        # fixed size snippets with progress prior
        rand_length = np.random.randint(min_snippet_length, max_snippet_length, size=num_snippets)
        for n in range(num_snippets):
            # pick two random demonstrations
            bi, bj = np.random.choice(num_bins, size=(2, ), replace=False)
            ti = np.random.choice(len(self.pre_expert_data[bi]))
            tj = np.random.choice(len(self.pre_expert_data[bj]))
            # create random snippets
            # find min length of both demos to ensure we can pick a demo no earlier
            # than that chosen in worse preferred demo
            min_length = min(len(self.pre_expert_data[bi][ti]), len(self.pre_expert_data[bj][tj]))
            if bi < bj:  # pick tj snippet to be later than ti
                ti_start = np.random.randint(min_length - rand_length[n] + 1)
                # print(ti_start, len(demonstrations[tj]))
                tj_start = np.random.randint(ti_start, len(self.pre_expert_data[bj][tj]) - rand_length[n] + 1)
            else:  # ti is better so pick later snippet in ti
                tj_start = np.random.randint(min_length - rand_length[n] + 1)
                # print(tj_start, len(demonstrations[ti]))
                ti_start = np.random.randint(tj_start, len(self.pre_expert_data[bi][ti]) - rand_length[n] + 1)
            # skip everyother framestack to reduce size
            traj_i = self.pre_expert_data[bi][ti][ti_start:ti_start + rand_length[n]:2]
            traj_j = self.pre_expert_data[bj][tj][tj_start:tj_start + rand_length[n]:2]

            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
            label = int(bi <= bj)
            self.training_obs.append((traj_i, traj_j))
            self.training_labels.append(label)
        self._logger.info(("maximum traj length: {}".format(max_traj_length)))
        return self.training_obs, self.training_labels

    def _train(self):
        # check if gpu available
        device = self.device  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assume that we are on a CUDA machine, then this should print a CUDA device:
        self._logger.info("device: {}".format(device))
        training_inputs, training_outputs = self.training_obs, self.training_labels
        loss_criterion = nn.CrossEntropyLoss()

        cum_loss = 0.0
        training_data = list(zip(training_inputs, training_outputs))
        for epoch in range(self.cfg.reward_model.update_per_collect):  # todo
            np.random.shuffle(training_data)
            training_obs, training_labels = zip(*training_data)
            for i in range(len(training_labels)):

                # traj_i, traj_j has the same length, however, they change as i increases
                traj_i, traj_j = training_obs[i]  # traj_i is a list of array generated by env.step
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(device)
                traj_j = torch.from_numpy(traj_j).float().to(device)

                # training_labels[i] is a boolean integer: 0 or 1
                labels = torch.tensor([training_labels[i]]).to(device)

                # forward + backward + zero out gradient + optimize
                outputs, abs_rewards = self.reward_model.forward(traj_i, traj_j)
                outputs = outputs.unsqueeze(0)
                loss = loss_criterion(outputs, labels) + self.l1_reg * abs_rewards
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # print stats to see if learning
                item_loss = loss.item()
                cum_loss += item_loss
                if i % 100 == 99:
                    self._logger.info("epoch {}:{} loss {}".format(epoch, i, cum_loss))
                    self._logger.info("abs_returns: {}".format(abs_rewards))
                    cum_loss = 0.0
                    self._logger.info("check pointing")
                    torch.save(self.reward_model.state_dict(), self.cfg.reward_model.reward_model_path)
        torch.save(self.reward_model.state_dict(), self.cfg.reward_model.reward_model_path)
        self._logger.info("finished training")

    def train(self):
        self._train()
        # print out predicted cumulative returns and actual returns
        sorted_returns = sorted(self.learning_returns, key=lambda s: s[0])
        demonstrations = [
            x for _, x in sorted(zip(self.learning_returns, self.pre_expert_data), key=lambda pair: pair[0][0])
        ]
        with torch.no_grad():
            pred_returns = [self.predict_traj_return(self.reward_model, traj[0]) for traj in demonstrations]
        for i, p in enumerate(pred_returns):
            self._logger.info("{} {} {}".format(i, p, sorted_returns[i][0]))
        info = {
            "demo_length": [len(d[0]) for d in self.pre_expert_data],
            "min_snippet_length": self.min_snippet_length,
            "max_snippet_length": min(np.min([len(d[0]) for d in self.pre_expert_data]), self.max_snippet_length),
            "len_num_training_obs": len(self.training_obs),
            "lem_num_labels": len(self.training_labels),
            "accuracy": self.calc_accuracy(self.reward_model, self.training_obs, self.training_labels),
        }
        self._logger.info(
            "accuracy and comparison:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
        )

    def predict_traj_return(self, net, traj):
        device = self.device
        # torch.set_printoptions(precision=20)
        # torch.use_deterministic_algorithms(True)
        with torch.no_grad():
            rewards_from_obs = net.cum_return(
                torch.from_numpy(np.array(traj)).float().to(device), mode='batch'
            )[0].squeeze().tolist()
            # rewards_from_obs1 = net.cum_return(torch.from_numpy(np.array([traj[0]])).float().to(device))[0].item()
            # different precision
        return sum(rewards_from_obs)  # rewards_from_obs is a list of floats

    def calc_accuracy(self, reward_network, training_inputs, training_outputs):
        device = self.device
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        with torch.no_grad():
            for i in range(len(training_inputs)):
                label = training_outputs[i]
                traj_i, traj_j = training_inputs[i]
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(device)
                traj_j = torch.from_numpy(traj_j).float().to(device)

                #forward to get logits
                outputs, abs_return = reward_network.forward(traj_i, traj_j)
                _, pred_label = torch.max(outputs, 0)
                if pred_label.item() == label:
                    num_correct += 1.
        return num_correct / len(training_inputs)

    def pred_data(self, data):
        obs = [default_collate(data[i])['obs'] for i in range(len(data))]
        res = [torch.sum(default_collate(data[i])['reward']).item() for i in range(len(data))]
        pred_returns = [self.predict_traj_return(self.reward_model, obs[i]) for i in range(len(obs))]
        return {'real': res, 'pred': pred_returns}

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

        res = collect_states(train_data_augmented)
        res = torch.stack(res).to(self.device)
        with torch.no_grad():
            sum_rewards, sum_abs_rewards = self.reward_model.cum_return(res, mode='batch')

        for item, rew in zip(train_data_augmented, sum_rewards):  # TODO optimise this loop as well ?
            item['reward'] = rew

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
        pass

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.training_obs.clear()
        self.training_labels.clear()

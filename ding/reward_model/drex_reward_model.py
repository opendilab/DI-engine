import copy

from easydict import EasyDict
import numpy as np
import pickle

import torch
import torch.nn as nn

from ding.utils import REWARD_MODEL_REGISTRY
from ding.utils.data import default_collate

from .trex_reward_model import TrexRewardModel


@REWARD_MODEL_REGISTRY.register('drex')
class DrexRewardModel(TrexRewardModel):
    config = dict(
        type='drex',
        learning_rate=1e-5,
        update_per_collect=100,
        batch_size=64,
        target_new_data_count=64,
        hidden_size=128,
        num_trajs=0,  # number of downsampled full trajectories
        num_snippets=6000,  # number of short subtrajectories to sample
    )

    bc_cfg = None

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`EasyDict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`SummaryWriter`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(DrexRewardModel, self).__init__(copy.deepcopy(config), device, tb_logger)

        self.pre_expert_data = []
        self.train_data = []

        self.created_data = []
        self.created_data_returns = []

        self.load_expert_data()

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config.expert_data_path`` attribute in self
        Effects:
            This is a side effect function which updates the expert data attribute \
                (i.e. ``self.expert_data``) with ``fn:concat_state_action_pairs``
        """
        with open(self.cfg.reward_model.offline_data_path + '/created_data.pkl', 'rb') as f:
            self.created_data = pickle.load(f)
        with open(self.cfg.reward_model.offline_data_path + '/created_data_returns.pkl', 'rb') as f:
            self.created_data_returns = pickle.load(f)
        with open(self.cfg.reward_model.offline_data_path + '/suboptimal_data.pkl', 'rb') as f:
            self.pre_expert_data = pickle.load(f)
        self.create_training_data()

        self._logger.info("num_training_obs: {}".format(len(self.training_obs)))
        self._logger.info("num_labels: {}".format(len(self.training_labels)))

    def create_training_data(self):
        num_trajs = self.num_trajs
        num_snippets = self.num_snippets
        min_snippet_length = self.min_snippet_length
        max_snippet_length = self.max_snippet_length

        demo_lengths = []
        for i in range(len(self.created_data)):
            demo_lengths.append([len(d) for d in self.created_data[i]])

        self._logger.info("demo_lengths: {}".format(demo_lengths))
        max_snippet_length = min(np.min(demo_lengths), max_snippet_length)
        self._logger.info("min snippet length: {}".format(min_snippet_length))
        self._logger.info("max snippet length: {}".format(max_snippet_length))

        # sorted_returns = sorted(self.learning_returns)
        # self._logger.info("sorted learning returns: {}".format(sorted_returns))

        # collect training data
        max_traj_length = 0
        num_bins = len(self.created_data)
        assert num_bins >= 2

        # add full trajs (for use on Enduro)
        si = np.random.randint(6, size=num_trajs)
        sj = np.random.randint(6, size=num_trajs)
        step = np.random.randint(3, 7, size=num_trajs)
        for n in range(num_trajs):
            # pick two random demonstrations
            bi, bj = np.random.choice(num_bins, size=(2,), replace=False)
            ti = np.random.choice(len(self.created_data[bi]))
            tj = np.random.choice(len(self.created_data[bj]))
            # create random partial trajs by finding random start frame and random skip frame
            traj_i = self.created_data[bi][ti][si[n]::step[n]]  # slice(start,stop,step)
            traj_j = self.created_data[bj][tj][sj[n]::step[n]]

            label = int(bi <= bj)

            self.training_obs.append((traj_i, traj_j))
            self.training_labels.append(label)
            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

        # fixed size snippets with progress prior
        rand_length = np.random.randint(min_snippet_length, max_snippet_length, size=num_snippets)
        for n in range(num_snippets):
            # pick two random demonstrations
            bi, bj = np.random.choice(num_bins, size=(2,), replace=False)
            ti = np.random.choice(len(self.created_data[bi]))
            tj = np.random.choice(len(self.created_data[bj]))
            # create random snippets
            # find min length of both demos to ensure we can pick a demo no earlier
            # than that chosen in worse preferred demo
            min_length = min(len(self.created_data[bi][ti]), len(self.created_data[bj][tj]))
            if bi < bj:  # pick tj snippet to be later than ti
                ti_start = np.random.randint(min_length - rand_length[n] + 1)
                # print(ti_start, len(demonstrations[tj]))
                tj_start = np.random.randint(ti_start, len(self.created_data[bj][tj]) - rand_length[n] + 1)
            else:  # ti is better so pick later snippet in ti
                tj_start = np.random.randint(min_length - rand_length[n] + 1)
                # print(tj_start, len(demonstrations[ti]))
                ti_start = np.random.randint(tj_start, len(self.created_data[bi][ti]) - rand_length[n] + 1)
            # skip everyother framestack to reduce size
            traj_i = self.created_data[bi][ti][ti_start:ti_start + rand_length[n]:2]
            traj_j = self.created_data[bj][tj][tj_start:tj_start + rand_length[n]:2]

            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
            label = int(bi <= bj)
            self.training_obs.append((traj_i, traj_j))
            self.training_labels.append(label)
        self._logger.info(("maximum traj length: {}".format(max_traj_length)))
        return self.training_obs, self.training_labels

    def train(self):
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

        return_dict = self.pred_data(self.pre_expert_data)
        res, pred_returns = return_dict['real'], return_dict['pred']
        self._logger.info("real: " + str(res))
        self._logger.info("pred: " + str(pred_returns))

        info = {
            "min_snippet_length": self.min_snippet_length,
            "max_snippet_length": self.max_snippet_length,
            "len_num_training_obs": len(self.training_obs),
            "lem_num_labels": len(self.training_labels),
            "accuracy": self.calc_accuracy(self.reward_model, self.training_obs, self.training_labels),
        }
        self._logger.info(
            "accuracy and comparison:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
        )

    def pred_data(self, data):
        obs = [default_collate(data[i])['obs'] for i in range(len(data))]
        res = [torch.sum(default_collate(data[i])['reward']).item() for i in range(len(data))]
        pred_returns = [self.predict_traj_return(self.reward_model, obs[i]) for i in range(len(obs))]
        return {'real': res, 'pred': pred_returns}


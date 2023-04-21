import copy
import random

import numpy as np
import torch
import torch.optim as optim
from easydict import EasyDict

from ding.utils import RunningMeanStd
from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel
from .network import RNDNetwork, InverseNetwork


def collect_data_and_exclude_null_data_rnd(data_in):
    res = []
    for item in data_in:
        if torch.nonzero(torch.tensor(item['null']).float()).shape[0] != 0:  # if have null padding in data
            # the index of not null data in data_in
            # not_null_index = torch.nonzero(torch.tensor(item['null']).float()).squeeze(-1)
            null_start_index = int(torch.nonzero(torch.tensor(item['null']).float()).squeeze(-1)[0])
            obs = item['obs'][:null_start_index]  # exclude the null padding data
        else:
            obs = item['obs']  # sequence data
        res.append(obs)
    return res


def collect_data_rnd(data_in):
    res = []
    is_null_list = []
    for item in data_in:
        state = item['obs']
        is_null = item['null']
        res.append(state)
        is_null_list.append(is_null)
    return res, is_null_list


def collect_data_and_exclude_null_data_episodic(data_in):
    obs_list = []
    action_list = []
    for item in data_in:
        if torch.nonzero(torch.tensor(item['null']).float()).shape[0] != 0:  # if have null padding in data
            # the index of not null data in data_in
            # not_null_index = torch.nonzero(torch.tensor(item['null']).float()).squeeze(-1)
            null_start_index = int(torch.nonzero(torch.tensor(item['null']).float()).squeeze(-1)[0])
            obs = item['obs'][:null_start_index]  # sequence data
            action = item['action'][:null_start_index]  # exclude the null padding data
        else:
            obs = item['obs']  # sequence data
            action = item['action']
        obs_list.append(obs)
        action_list.append(action)
    return obs_list, action_list


def collect_data_episodic(data_in):
    res = []
    is_null_list = []
    for item in data_in:
        state = item['obs']
        is_null = item['null']
        res.append(state)
        is_null_list.append(is_null)
    return res, is_null_list


@REWARD_MODEL_REGISTRY.register('rnd-ngu')
class RndNGURewardModel(BaseRewardModel):
    r"""
    Overview:
        inter-episodic/RND reward model for NGU.
        The corresponding paper is `never give up: learning directed exploration strategies`.
    """
    config = dict(
        type='rnd-ngu',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(RndNGURewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = RNDNetwork(config.obs_shape, config.hidden_size_list)
        self.reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_data_total = []
        self.train_data = []
        self.opt = optim.Adam(self.reward_model.predictor.parameters(), config.learning_rate)
        self.train_cnt_icm = 0
        self.estimate_cnt_rnd = 0
        self._running_mean_std_rnd = RunningMeanStd(epsilon=1e-4)
        self.only_use_last_five_frames = config.only_use_last_five_frames_for_icm_rnd

    def _train(self) -> torch.Tensor:
        train_data: list = random.sample(list(self.train_data_cur), self.cfg.batch_size)

        train_data: torch.Tensor = torch.stack(train_data).to(self.device)

        loss = self.reward_model.learn(train_data)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss

    def train(self) -> None:
        if self.only_use_last_five_frames:
            # self.train_obs shape list(list) [batch_size,seq_length,N

            # stack episode dim
            self.train_obs = [torch.stack(episode_obs[-5:], dim=0) for episode_obs in self.train_data_total]

            # stack batch dim
            # way 1
            if isinstance(self.cfg.obs_shape, int):
                self.train_data_cur = torch.stack(
                    self.train_obs, dim=0
                ).view(len(self.train_obs) * len(self.train_obs[0]), self.cfg.obs_shape)
            else:  # len(self.cfg.obs_shape) == 3 for image obs
                self.train_data_cur = torch.stack(
                    self.train_obs, dim=0
                ).view(len(self.train_obs) * self.train_obs[0].shape[0], *self.cfg.obs_shape)
            # way 2
            # self.train_data_cur = torch.cat(self.train_obs, 0)

        else:
            self.train_data_cur = sum(self.train_data_total, [])
            # another implementation way
            # tmp = []
            # for i in range(len(self.train_data)):
            #     tmp += self.train_data[i]
            # self.train_data = tmp

        for _ in range(self.cfg.update_per_collect):
            loss = self._train()
            self.tb_logger.add_scalar('rnd_reward/loss', loss, self.train_cnt_icm)
            self.train_cnt_icm += 1

    def estimate(self, data: list) -> torch.Tensor:
        """
        Rewrite the reward key in each row of the data.
        """
        obs, is_null = collect_data_rnd(data)
        if isinstance(obs[0], list):  # if obs shape list( list(torch.tensor) )
            obs = sum(obs, [])

        obs = torch.stack(obs).to(self.device)

        with torch.no_grad():
            reward = self.reward_model.forward(obs, norm=False)
            self._running_mean_std_rnd.update(reward.cpu().numpy())
            # transform to mean 1 std 1
            reward = 1 + (reward - self._running_mean_std_rnd.mean) / (self._running_mean_std_rnd.std + 1e-11)
            self.estimate_cnt_rnd += 1
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_max', reward.max(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_mean', reward.mean(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_min', reward.min(), self.estimate_cnt_rnd)
        return reward

    def collect_data(self, data: list) -> None:
        self.train_data_total.extend(collect_data_and_exclude_null_data_rnd(data))

    def clear_data(self) -> None:
        self.train_data_total.clear()

    def reward_deepcopy(self, train_data):
        """
        this method deepcopy reward part in train_data, and other parts keep shallow copy
        to avoid the reward part of train_data in the replay buffer be incorrectly modified.
        """
        train_data_reward_deepcopy = [
            {k: copy.deepcopy(v) if k == 'reward' else v
             for k, v in sample.items()} for sample in train_data
        ]
        return train_data_reward_deepcopy


@REWARD_MODEL_REGISTRY.register('episodic')
class EpisodicNGURewardModel(BaseRewardModel):
    r"""
    Overview:
        Episodic reward model for NGU.
        The corresponding paper is `never give up: learning directed exploration strategies`.
    """
    config = dict(
        type='episodic',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
        # means if using rescale trick to the last non-zero reward
        # when combing extrinsic and intrinsic reward.
        # the rescale trick only used in:
        # 1. sparse reward env minigrid, in which the last non-zero reward is a strong positive signal
        # 2. the last reward of each episode directly reflects the agent's completion of the task, e.g. lunarlander
        # Note that the ngu intrinsic reward is a positive value (max value is 5), in these envs,
        # the last non-zero reward should not be overwhelmed by intrinsic rewards, so we need rescale the
        # original last nonzero extrinsic reward.
        last_nonzero_reward_rescale=False,
        # means the rescale value for the last non-zero reward, only used when last_nonzero_reward_rescale is True
        last_nonzero_reward_weight=1,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(EpisodicNGURewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.episodic_reward_model = InverseNetwork(
            config.obs_shape, config.action_shape, config.hidden_size_list, self.device
        )
        self.episodic_reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_obs_total = []
        self.train_action_total = []
        self.opt = optim.Adam(self.episodic_reward_model.parameters(), config.learning_rate)
        self.estimate_cnt_episodic = 0
        self.train_cnt_episodic = 0
        self.only_use_last_five_frames = config.only_use_last_five_frames_for_icm_rnd

    def _train(self) -> torch.Tensor:
        # sample episode's timestep index
        train_index = np.random.randint(low=0, high=self.train_obs.shape[0], size=self.cfg.batch_size)

        train_obs: torch.Tensor = self.train_obs[train_index].to(self.device)  # shape (self.cfg.batch_size, obs_dim)
        train_next_obs: torch.Tensor = self.train_next_obs[train_index].to(self.device)
        train_action: torch.Tensor = self.train_action[train_index].to(self.device)

        train_data = {'obs': train_obs, 'next_obs': train_next_obs, 'action': train_action}
        inverse_loss = self.episodic_reward_model.learn(train_data)
        self.opt.zero_grad()
        inverse_loss.backward()
        self.opt.step()

        return inverse_loss

    def train(self) -> None:
        self.train_next_obs_total = copy.deepcopy(self.train_obs_total)

        if self.only_use_last_five_frames:
            # self.train_obs shape: list(list) [batch_size,seq_length,obs_dim]
            self.train_obs = [torch.stack(episode_obs[-6:-1], dim=0) for episode_obs in self.train_obs_total]
            self.train_next_obs = [torch.stack(episode_obs[-5:], dim=0) for episode_obs in self.train_next_obs_total]
            self.train_action = [
                torch.stack(episode_action[-6:-1], dim=0) for episode_action in self.train_action_total
            ]
        else:
            self.train_obs = [
                torch.stack(episode_obs[:-1], dim=0) for episode_obs in self.train_obs_total if len(episode_obs) > 1
            ]
            self.train_next_obs = [
                torch.stack(episode_next_obs[1:], dim=0) for episode_next_obs in self.train_next_obs_total
                if len(episode_next_obs) > 1
            ]
            self.train_action = [
                torch.stack(episode_action[:-1], dim=0) for episode_action in self.train_action_total
                if len(episode_action) > 1
            ]

        # stack batch dim
        self.train_obs = torch.cat(self.train_obs, 0)
        self.train_next_obs = torch.cat(self.train_next_obs, 0)
        self.train_action = torch.cat(self.train_action, 0)

        for _ in range(self.cfg.update_per_collect):
            loss = self._train()
            self.tb_logger.add_scalar('episodic_reward/train_loss', loss, self.train_cnt_episodic)
            self.train_cnt_episodic += 1

    def estimate(self, data: list) -> torch.Tensor:
        """
        Rewrite the reward key in each row of the data.
        """

        obs, is_null = collect_data_episodic(data)

        with torch.no_grad():
            episodic_reward, episodic_reward_real_mean = self.episodic_reward_model.forward(obs, is_null)
            self.estimate_cnt_episodic += 1

            self.tb_logger.add_scalar(
                'episodic_reward/episodic_reward_max', episodic_reward.max(), self.estimate_cnt_episodic
            )
            self.tb_logger.add_scalar(
                'episodic_reward/episodic_reward_mean', episodic_reward_real_mean, self.estimate_cnt_episodic
            )
            self.tb_logger.add_scalar(
                'episodic_reward/episodic_reward_min', episodic_reward.min(), self.estimate_cnt_episodic
            )
            self.tb_logger.add_scalar(
                'episodic_reward/episodic_reward_std_', episodic_reward.std(), self.estimate_cnt_episodic
            )
            # transform to [0,1]: er01
            episodic_reward = (episodic_reward -
                               episodic_reward.min()) / (episodic_reward.max() - episodic_reward.min() + 1e-11)
            """1. transform to batch mean1: erbm1"""
            # episodic_reward = episodic_reward / (episodic_reward.mean() + 1e-11)
            # the null_padding transition have episodic reward=0,
            # episodic_reward = episodic_reward / (episodic_reward_real_mean + 1e-11)
            """2. transform to long-term mean1: erlm1"""
            # episodic_reward = episodic_reward / self._running_mean_std_episodic_reward.mean
            """3. transform to mean 0, std 1, which is wrong, rnd_reward is in [1,5], episodic reward should >0,
            otherwise, e.g. when the  episodic_reward is -2, the rnd_reward larger,
            the total intrinsic reward smaller, which is not correct."""
            # episodic_reward = (episodic_reward - self._running_mean_std_episodic_reward.mean)
            # / self._running_mean_std_episodic_reward.std
            """4. transform to std1, which is not very meaningful"""
            # episodic_reward = episodic_reward / self._running_mean_std_episodic_reward.std

        return episodic_reward

    def collect_data(self, data: list) -> None:
        train_obs, train_action = collect_data_and_exclude_null_data_episodic(data)
        self.train_obs_total.extend(train_obs)
        self.train_action_total.extend(train_action)

    def clear_data(self) -> None:
        self.train_obs_total = []
        self.train_action_total = []

    def fusion_reward(
        self, train_data, inter_episodic_reward, episodic_reward, nstep, collector_env_num, tb_logger, estimate_cnt
    ):
        # NOTE: deepcopy reward part of train_data is very important,
        # otherwise the reward of train_data in the replay buffer will be incorrectly modified.
        data = self.reward_deepcopy(train_data)
        estimate_cnt += 1
        index_to_beta = {
            i: 0.3 * torch.sigmoid(torch.tensor(10 * (2 * i - (collector_env_num - 2)) / (collector_env_num - 2)))
            for i in range(collector_env_num)
        }
        batch_size = len(data)
        seq_length = len(data[0]['reward'])
        device = data[0]['reward'][0].device
        intrinsic_reward_type = 'add'
        intrisic_reward = episodic_reward * torch.clamp(inter_episodic_reward, min=1, max=5)
        tb_logger.add_scalar('intrinsic_reward/intrinsic_reward_max', intrisic_reward.max(), estimate_cnt)
        tb_logger.add_scalar('intrinsic_reward/intrinsic_reward_mean', intrisic_reward.mean(), estimate_cnt)
        tb_logger.add_scalar('intrinsic_reward/intrinsic_reward_min', intrisic_reward.min(), estimate_cnt)

        if not isinstance(data[0], (list, dict)):
            # not rnn based rl algorithm
            intrisic_reward = intrisic_reward.to(device)
            intrisic_reward = torch.chunk(intrisic_reward, intrisic_reward.shape[0], dim=0)
            for item, rew in zip(data, intrisic_reward):
                if intrinsic_reward_type == 'add':
                    item['reward'] += rew * index_to_beta[data['beta']]
        else:
            # rnn based rl algorithm
            intrisic_reward = intrisic_reward.to(device)

            # tensor to tuple
            intrisic_reward = torch.chunk(intrisic_reward, int(intrisic_reward.shape[0]), dim=0)

            if self.cfg.last_nonzero_reward_weight is None and self.cfg.last_nonzero_reward_rescale:
                # for minigrid env
                self.cfg.last_nonzero_reward_weight = seq_length

            # this is for the nstep rl algorithms
            for i in range(batch_size):  # batch_size typically 64
                for j in range(seq_length):  # burnin+unroll_len is the sequence length, e.g. 100=2+98
                    if j < seq_length - nstep:
                        intrinsic_reward = torch.cat(
                            [intrisic_reward[i * seq_length + j + k] for k in range(nstep)], dim=0
                        )
                        # if intrinsic_reward_type == 'add':
                        if not data[i]['null'][j]:
                            # if data[i]['null'][j]==True, means its's null data, only the not null data,
                            # we add a intrinsic_reward
                            if data[i]['done'][j] and self.cfg.last_nonzero_reward_rescale:
                                # if not null data, and data[i]['done'][j]==True, so this is the last nstep transition
                                # in the original data.

                                # means if using rescale trick to the last non-zero reward
                                # when combing extrinsic and intrinsic reward.
                                # only used in sparse reward env minigrid, in which the last non-zero reward
                                # is a strong positive signal, should not be overwhelmed by intrinsic rewards。
                                for k in reversed(range(nstep)):
                                    # here we want to find the last nonzero reward in the nstep reward list:
                                    # data[i]['reward'][j], that is also the last reward in the sequence, here,
                                    # we set the sequence length is large enough,
                                    # so we can consider the sequence as the whole episode plus null_padding

                                    # TODO(pu): what should we do if the last reward in the whole episode is zero?
                                    if data[i]['reward'][j][k] != 0:
                                        # find the last one that is nonzero, and enlarging <seq_length> times
                                        last_nonzero_rew = copy.deepcopy(data[i]['reward'][j][k])
                                        data[i]['reward'][j][k] = \
                                            self.cfg.last_nonzero_reward_weight * last_nonzero_rew + \
                                            intrinsic_reward[k] * index_to_beta[int(data[i]['beta'][j])]
                                        # substitute the kth reward in the list data[i]['reward'][j] with <seq_length>
                                        # times amplified reward
                                        break
                            else:
                                data[i]['reward'][j] = data[i]['reward'][j] + intrinsic_reward * index_to_beta[
                                    int(data[i]['beta'][j])]

        return data, estimate_cnt

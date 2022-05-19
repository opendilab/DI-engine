import copy
import random
from typing import Union, Tuple, Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydict import EasyDict

from ding.model import FCEncoder, ConvEncoder
from ding.utils import RunningMeanStd
from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


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


class RndNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType) -> None:
        super(RndNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.target = FCEncoder(obs_shape, hidden_size_list)
            self.predictor = FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.target = ConvEncoder(obs_shape, hidden_size_list)
            self.predictor = ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, "
                "please customize your own RND model".format(obs_shape)
            )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predict_feature, target_feature


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
        self.reward_model = RndNetwork(config.obs_shape, config.hidden_size_list)
        self.reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_data_total = []
        self.train_data = []
        self.opt = optim.Adam(self.reward_model.predictor.parameters(), config.learning_rate)
        self.estimate_cnt_rnd = 0
        self._running_mean_std_rnd = RunningMeanStd(epsilon=1e-4)
        self.only_use_last_five_frames = config.only_use_last_five_frames_for_icm_rnd

    def _train(self) -> None:
        train_data: list = random.sample(list(self.train_data_cur), self.cfg.batch_size)

        train_data: torch.Tensor = torch.stack(train_data).to(self.device)

        predict_feature, target_feature = self.reward_model(train_data)
        loss = F.mse_loss(predict_feature, target_feature.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

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
            self._train()

    def estimate(self, data: list) -> torch.Tensor:
        """
        Rewrite the reward key in each row of the data.
        """
        obs, is_null = collect_data_rnd(data)
        if isinstance(obs[0], list):  # if obs shape list( list(torch.tensor) )
            obs = sum(obs, [])

        obs = torch.stack(obs).to(self.device)

        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(obs)
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
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


class InverseNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], action_shape, hidden_size_list: SequenceType) -> None:
        super(InverseNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.embedding_net = FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.embedding_net = ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_size_list[-1] * 2, 512), nn.ReLU(inplace=True), nn.Linear(512, action_shape)
        )

    def forward(self, inputs: Dict, inference: bool = False) -> Dict:
        if inference:
            with torch.no_grad():
                cur_obs_embedding = self.embedding_net(inputs['obs'])
            return cur_obs_embedding
        else:
            # obs: torch.Tensor, next_obs: torch.Tensor
            cur_obs_embedding = self.embedding_net(inputs['obs'])
            next_obs_embedding = self.embedding_net(inputs['next_obs'])
            # get pred action
            obs_plus_next_obs = torch.cat([cur_obs_embedding, next_obs_embedding], dim=-1)
            pred_action_logits = self.inverse_net(obs_plus_next_obs)
            pred_action_probs = nn.Softmax(dim=-1)(pred_action_logits)
            return pred_action_logits, pred_action_probs


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
        self.episodic_reward_model = InverseNetwork(config.obs_shape, config.action_shape, config.hidden_size_list)
        self.episodic_reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_obs_total = []
        self.train_action_total = []
        self.opt = optim.Adam(self.episodic_reward_model.parameters(), config.learning_rate)
        self.estimate_cnt_episodic = 0
        self._running_mean_std_episodic_dist = RunningMeanStd(epsilon=1e-4)
        self._running_mean_std_episodic_reward = RunningMeanStd(epsilon=1e-4)
        self.only_use_last_five_frames = config.only_use_last_five_frames_for_icm_rnd

    def _train(self) -> None:
        # sample episode's timestep index
        train_index = np.random.randint(low=0, high=self.train_obs.shape[0], size=self.cfg.batch_size)

        train_obs: torch.Tensor = self.train_obs[train_index].to(self.device)  # shape (self.cfg.batch_size, obs_dim)
        train_next_obs: torch.Tensor = self.train_next_obs[train_index].to(self.device)
        train_action: torch.Tensor = self.train_action[train_index].to(self.device)

        train_data = {'obs': train_obs, 'next_obs': train_next_obs}
        pred_action_logits, pred_action_probs = self.episodic_reward_model(train_data)

        inverse_loss = F.cross_entropy(pred_action_logits, train_action.squeeze(-1))
        self.opt.zero_grad()
        inverse_loss.backward()
        self.opt.step()

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
            self._train()

    def _compute_intrinsic_reward(
            self,
            episodic_memory: List,
            current_controllable_state: torch.Tensor,
            k=10,
            kernel_cluster_distance=0.008,
            kernel_epsilon=0.0001,
            c=0.001,
            siminarity_max=8,
    ) -> torch.Tensor:
        # this function is modified from https://github.com/Coac/never-give-up/blob/main/embedding_model.py
        state_dist = torch.cdist(current_controllable_state.unsqueeze(0), episodic_memory, p=2).squeeze(0).sort()[0][:k]
        self._running_mean_std_episodic_dist.update(state_dist.cpu().numpy())
        state_dist = state_dist / (self._running_mean_std_episodic_dist.mean + 1e-11)

        state_dist = torch.clamp(state_dist - kernel_cluster_distance, min=0, max=None)
        kernel = kernel_epsilon / (state_dist + kernel_epsilon)
        s = torch.sqrt(torch.clamp(torch.sum(kernel), min=0, max=None)) + c

        if s > siminarity_max:
            print('s > siminarity_max:', s.max(), s.min())
            return torch.tensor(0)  # NOTE
        return 1 / s
        # average value 1/( ( 10* 1e-4/(1+1e-4) )**(1/2)+1e-3 ) = 30

    def estimate(self, data: list) -> torch.Tensor:
        """
        Rewrite the reward key in each row of the data.
        """

        obs, is_null = collect_data_episodic(data)
        # obs shape list(list()) [batch_size,seq_length,obs_dim]
        batch_size = len(obs)
        seq_length = len(obs[0])

        # stack episode dim
        obs = [torch.stack(episode_obs, dim=0) for episode_obs in obs]

        # stack batch dim
        # way 0
        if isinstance(self.cfg.obs_shape, int):
            obs = torch.stack(obs, dim=0).view(batch_size * seq_length, self.cfg.obs_shape).to(self.device)
        else:  # len(self.cfg.obs_shape) == 3 for image obs
            obs = torch.stack(obs, dim=0).view(batch_size * seq_length, *self.cfg.obs_shape).to(self.device)
        # way 2
        # obs = torch.cat(obs, 0)

        inputs = {'obs': obs, 'is_null': is_null}
        with torch.no_grad():
            cur_obs_embedding = self.episodic_reward_model(inputs, inference=True)
            cur_obs_embedding = cur_obs_embedding.view(batch_size, seq_length, -1)
            episodic_reward = [[] for _ in range(batch_size)]
            null_cnt = 0  # the number of null transitions in the whole minibatch
            for i in range(batch_size):
                for j in range(seq_length):
                    if j < 10:
                        # if self._running_mean_std_episodic_reward.mean is not None:
                        #     episodic_reward[i].append(torch.tensor(self._running_mean_std_episodic_reward.mean).to(self.device))
                        # else:
                        episodic_reward[i].append(torch.tensor(0.).to(self.device))
                    elif j:
                        episodic_memory = cur_obs_embedding[i][:j]
                        reward = self._compute_intrinsic_reward(episodic_memory,
                                                                cur_obs_embedding[i][j]).to(self.device)
                        episodic_reward[i].append(reward)

                if torch.nonzero(torch.tensor(is_null[i]).float()).shape[0] != 0:
                    # TODO(pu): if have null padding, the episodic_reward should be 0
                    not_null_index = torch.nonzero(torch.tensor(is_null[i]).float()).squeeze(-1)
                    null_start_index = int(torch.nonzero(torch.tensor(is_null[i]).float()).squeeze(-1)[0])
                    # add the number of null transitions in i'th sequence in batch
                    null_cnt = null_cnt + seq_length - null_start_index
                    for k in range(null_start_index, seq_length):
                        episodic_reward[i][k] = torch.tensor(0).to(self.device)
                        # episodic_reward[i][null_start_index:-1]=[torch.tensor(0).to(self.device)
                        # for i in range(seq_length-null_start_index)]

            # list(list(tensor)) -> tensor
            tmp = [torch.stack(episodic_reward_tmp, dim=0) for episodic_reward_tmp in episodic_reward]
            # stack batch dim
            episodic_reward = torch.stack(tmp, dim=0)  # TODO(pu): image case
            episodic_reward = episodic_reward.view(-1)  # torch.Size([32, 42]) -> torch.Size([32*42]

            episodic_reward_real_mean = sum(episodic_reward) / (
                batch_size * seq_length - null_cnt
            )  # TODO(pu): recompute mean
            self.estimate_cnt_episodic += 1
            self._running_mean_std_episodic_reward.update(episodic_reward.cpu().numpy())

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

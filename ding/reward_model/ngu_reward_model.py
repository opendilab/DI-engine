import copy
import random
from typing import Tuple
from typing import Union, Dict, List

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


def collect_states(iterator):  # get total_states
    res = []
    for item in iterator:
        state = item['obs']
        res.append(state)
    return res


def collect_states_episodic(iterator):  # get total_states list(dict, dict,...)
    obs_list = []
    action_list = []
    for item in iterator:
        obs = item['obs']  # episode
        action = item['action']
        obs_list.append(obs)
        action_list.append(action)
    return obs_list, action_list


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
        # for param in self.target.parameters():
        #     param.requires_grad = False

    def forward(self, inputs: Dict, inference: bool = False) -> Dict:
        if inference:
            with torch.no_grad():
                cur_obs_embedding = self.embedding_net(inputs['obs'])
            return cur_obs_embedding
        else:
            # obs: torch.Tensor,next_obs: torch.Tensor
            cur_obs_embedding = self.embedding_net(inputs['obs'])
            next_obs_embedding = self.embedding_net(inputs['next_obs'])
            # get pred action
            obs_plus_next_obs = torch.cat([cur_obs_embedding, next_obs_embedding], dim=-1)
            pred_action_logits = self.inverse_net(obs_plus_next_obs)
            pred_action_probs = nn.Softmax(dim=-1)(pred_action_logits)
            return pred_action_logits, pred_action_probs


@REWARD_MODEL_REGISTRY.register('episodic')
class EpisodicRewardModel(BaseRewardModel):
    config = dict(
        type='episodic',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        # obs_shape=6,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(EpisodicRewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.episodic_reward_model = InverseNetwork(config.obs_shape, config.action_shape, config.hidden_size_list)
        self.episodic_reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        # self.train_data = []
        self.train_obs = []
        self.train_next_obs = []
        self.train_action = []
        self.opt = optim.Adam(self.episodic_reward_model.parameters(), config.learning_rate)
        self.estimate_cnt_episodic = 0

    def _train(self) -> None:
        # sample episode's timestep index
        train_index = np.random.randint(
            low=0, high=self.train_obs.shape[0], size=64
        )  # self.cfg.reward_model_batch_size)

        train_obs: torch.Tensor = self.train_obs[train_index].to(self.device)  # shape reward_model_batch_size, obs_dim
        train_next_obs: torch.Tensor = self.train_next_obs[train_index].to(self.device)
        train_action: torch.Tensor = self.train_action[train_index].to(self.device)

        train_data = {'obs': train_obs, 'next_obs': train_next_obs}
        pred_action_logits, pred_action_probs = self.episodic_reward_model(train_data)

        inverse_loss = F.cross_entropy(pred_action_logits, train_action.squeeze(-1))  # .mean(dim=1)
        self.opt.zero_grad()
        inverse_loss.backward()
        self.opt.step()

    def train(self) -> None:
        self._running_mean_std_episodic_dist = RunningMeanStd(epsilon=1e-4)
        self._running_mean_std_episodic_reward = RunningMeanStd(epsilon=1e-4)
        # stack episode dim
        self.train_next_obs = copy.deepcopy(self.train_obs)
        self.train_obs = [
            torch.stack(episode_obs[:-1], dim=0) for episode_obs in self.train_obs
        ]  # self.train_obs list(list) 32,42 batch_size,timesteps,dim
        self.train_next_obs = [torch.stack(episode_obs[1:], dim=0) for episode_obs in self.train_next_obs]
        self.train_action = [torch.stack(episode_action[:-1], dim=0) for episode_action in self.train_action]

        # stack batch dim
        if isinstance(self.cfg.obs_shape, int):
            self.train_obs = torch.stack(
                self.train_obs, dim=0
            ).view(len(self.train_obs) * len(self.train_obs[0]), self.cfg.obs_shape)  # -1) TODO image
        else:  #:len(self.cfg.obs_shape) == 3
            self.train_obs = torch.stack(
                self.train_obs, dim=0
            ).view(self.train_obs.shape[0] * self.train_obs.shape[1], *self.cfg.obs_shape)  # -1) TODO image

        self.train_next_obs = torch.stack(
            self.train_next_obs, dim=0
        ).view(len(self.train_next_obs) * len(self.train_next_obs[0]), -1)
        self.train_action = torch.stack(
            self.train_action, dim=0
        ).view(len(self.train_action) * len(self.train_action[0]), -1)
        # train_obs: list = random.sample(self.train_obs, self.cfg.batch_size)
        # train_obs: list = random.sample(self.train_data, self.cfg.batch_size)
        for _ in range(self.cfg.update_per_collect):  # self.cfg.update_per_collect_intrinsic_reward
            self._train()
        self.clear_data()

    # def _compute_intrinsic_reward(
    #         self,
    #         episodic_memory: List,
    #         current_controllable_state: torch.Tensor,
    #         k=10,
    #         kernel_cluster_distance=0.008,
    #         kernel_epsilon=0.001,
    #         c=0.001,
    #         sm=8,
    # ) -> torch.Tensor:  #kernel_epsilon=0.0001
    #     # this function is modified from https://github.com/Coac/never-give-up/blob/main/embedding_model.py
    #     state_dist = [(c_state, torch.dist(c_state, current_controllable_state)) for c_state in episodic_memory]
    #     state_dist.sort(key=lambda x: x[1])
    #     state_dist = state_dist[:k]
    #     dist = [d[1].item() for d in state_dist]
    #     dist = np.array(dist)

    #     self._running_mean_std_episodic_dist.update(dist)  #.cpu().numpy() # TODO
    #     dist = dist / self._running_mean_std_episodic_dist.mean  # TODO

    #     # dist = np.max(dist - kernel_cluster_distance, 0) #TODO

    #     kernel = kernel_epsilon / (dist + kernel_epsilon)
    #     s = np.sqrt(np.clip(np.sum(kernel), 0, None)) + c

    #     if np.isnan(s) or s > sm:
    #         print('np.isnan(s) or s > sm!:',s.max(),s.min())
    #         return torch.tensor(0)  # todo
    #     return torch.tensor(1 / s)
    def _compute_intrinsic_reward(
            self,
            episodic_memory: List,
            current_controllable_state: torch.Tensor,
            k=10,
            kernel_cluster_distance=0.008,
            kernel_epsilon=0.001,
            c=0.001,
            siminarity_max=8,
    ) -> torch.Tensor:  # kernel_epsilon=0.0001 # this function is modified from https://github.com/Coac/never-give-up/blob/main/embedding_model.py
        state_dist = torch.cdist(current_controllable_state.unsqueeze(0), episodic_memory, p=2).squeeze(0).sort()[0][:k]
        self._running_mean_std_episodic_dist.update(state_dist.cpu().numpy())  # TODO
        state_dist = state_dist / (self._running_mean_std_episodic_dist.mean + 1e-11)  # TODO

        # dist = np.max(dist - kernel_cluster_distance, 0) #TODO
        kernel = kernel_epsilon / (state_dist + kernel_epsilon)
        s = torch.sqrt(torch.clamp(torch.sum(kernel), min=0, max=None)) + c

        if s > siminarity_max:
            print('s > siminarity_max:', s.max(), s.min())
            return torch.tensor(0)  # todo
        if torch.isnan(s):
            print('torch.isnan(s):', s.max(), s.min())
            return torch.tensor(0)  # todo
        return 1 / s  # torch.tensor(1 / s)

    def estimate(self, data: list) -> None:
        """
        Rewrite the reward key in each row of the data.
        """
        obs = collect_states(data)  # list(list()) 32,42,obs_dim
        batch_size = len(obs)
        timesteps = len(obs[0])
        # stack episode dim
        obs = [torch.stack(episode_obs, dim=0) for episode_obs in obs]
        # stack batch dim
        if isinstance(self.cfg.obs_shape, int):
            obs = torch.stack(
                obs, dim=0
            ).view(batch_size * timesteps, self.cfg.obs_shape).to(self.device)  # -1) TODO image
        else:  # len(self.cfg.obs_shape) == 3
            obs = torch.stack(
                obs, dim=0
            ).view(batch_size * timesteps, *self.cfg.obs_shape).to(self.device)  # -1) TODO image
        # if isinstance(obs[0], list):  # if self.train_data list( list(torch.tensor) )
        #     tmp = []
        #     for i in range(len(data)):
        #         tmp += obs[i]
        #     obs = tmp
        # obs = torch.stack(obs).to(self.device)
        inputs = {'obs': obs}
        with torch.no_grad():
            cur_obs_embedding = self.episodic_reward_model(inputs, inference=True)
            cur_obs_embedding = cur_obs_embedding.view(batch_size, timesteps, -1)  # 32 42,64
            episodic_reward = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for j in range(timesteps):
                    if j <= 10:
                        # if self._running_mean_std_episodic.mean is not None:
                        #     episodic_reward[i].append(torch.tensor( self._running_mean_std_episodic.mean))
                        # else:
                        episodic_reward[i].append(torch.tensor(0.))
                    else:
                        episodic_memory = cur_obs_embedding[i][:j]
                        reward = self._compute_intrinsic_reward(episodic_memory,
                                                                cur_obs_embedding[i][j]).to(self.device)
                        episodic_reward[i].append(reward)

            # 32,42,1  list(list(tensor)) - > tensor
            tmp = [torch.stack(episodic_reward_tmp, dim=0) for episodic_reward_tmp in episodic_reward]
            # stack batch dim
            episodic_reward = torch.stack(tmp, dim=0)  # -1) TODO image
            episodic_reward = episodic_reward.view(-1)  # torch.Size([32, 42]) -> torch.Size([32*42]

            self.estimate_cnt_episodic += 1
            self.tb_logger.add_scalar(
                'episodic_reward/episodic_reward_max', episodic_reward.max(), self.estimate_cnt_episodic
            )
            self.tb_logger.add_scalar(
                'episodic_reward/episodic_reward_mean', episodic_reward.mean(), self.estimate_cnt_episodic
            )
            self.tb_logger.add_scalar(
                'episodic_reward/episodic_reward_min', episodic_reward.min(), self.estimate_cnt_episodic
            )
            # episodic_reward = (episodic_reward - episodic_reward.min()) / (episodic_reward.max() - episodic_reward.min() + 1e-8)
            # self._running_mean_std_episodic_reward.update(episodic_reward.cpu().numpy()) #.cpu().numpy() # TODO
            # episodic_reward =  episodic_reward / self._running_mean_std_episodic_reward.mean  # TODO
            episodic_reward = (episodic_reward - episodic_reward.min()) / (
                episodic_reward.max() - episodic_reward.min() + 1e-8
            )  # normalize to [0,1]
        return episodic_reward

    def collect_data(self, data: list) -> None:
        train_obs, train_action = collect_states_episodic(data)
        self.train_obs.extend(train_obs)
        self.train_action.extend(train_action)

    def clear_data(self) -> None:
        self.train_obs = []
        self.train_next_obs = []
        self.train_action = []


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
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predict_feature, target_feature


@REWARD_MODEL_REGISTRY.register('rnd')
class RndRewardModel(BaseRewardModel):
    config = dict(
        type='rnd',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        # obs_shape=6,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(RndRewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = RndNetwork(config.obs_shape, config.hidden_size_list)
        self.reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_data = []
        self.opt = optim.Adam(self.reward_model.predictor.parameters(), config.learning_rate)
        self.estimate_cnt_rnd = 0

    def _train(self) -> None:
        if isinstance(self.train_data[0], list):  # if self.train_data list( list(torch.tensor) ) rnn
            # tmp = []
            # for i in range(len(self.train_data)):
            #     tmp += self.train_data[i]
            # self.train_data = tmp
            self.train_data = sum(self.train_data, [])
        train_data: list = random.sample(self.train_data, self.cfg.batch_size)
        train_data: torch.Tensor = torch.stack(train_data).to(self.device)
        predict_feature, target_feature = self.reward_model(train_data)
        loss = F.mse_loss(predict_feature, target_feature.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        self._running_mean_std_rnd = RunningMeanStd(epsilon=1e-4)
        for _ in range(self.cfg.update_per_collect):
            self._train()
        self.clear_data()

    def estimate(self, data: list) -> None:
        """
        Rewrite the reward key in each row of the data.
        """
        obs = collect_states(data)
        if isinstance(obs[0], list):  # if self.train_data list( list(torch.tensor) )
            obs = sum(obs, [])

        obs = torch.stack(obs).to(self.device)
        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(obs)
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            self._running_mean_std_rnd.update(reward.cpu().numpy())
            reward = 1 + (reward - self._running_mean_std_rnd.mean) / self._running_mean_std_rnd.std  # TODO
            self.estimate_cnt_rnd += 1
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_max', reward.max(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_mean', reward.mean(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_min', reward.min(), self.estimate_cnt_rnd)
            # reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-8)
        return reward  # torch.Size([1344])

    def collect_data(self, data: list) -> None:
        self.train_data.extend(collect_states(data))

    def clear_data(self) -> None:
        self.train_data.clear()


def fusion_reward(data, inter_episodic_reward, episodic_reward, nstep, collector_env_num, tb_logger, estimate_cnt):
    estimate_cnt += 1
    # index_to_eps = {i: 0.4 ** (1 + 8 * i / (self._env_num - 1)) for i in range(self._env_num)}
    index_to_beta = {
        i: 0.3 * torch.sigmoid(torch.tensor(10 * (2 * i - (collector_env_num - 2)) / (collector_env_num - 2)))
        for i in range(collector_env_num)
    }
    index_to_gamma = {
        i: 1 - torch.exp(
            ((collector_env_num - 1 - i) * torch.log(torch.tensor(1 - 0.997)) + i * torch.log(torch.tensor(1 - 0.99))) /
            (collector_env_num - 1)
        )
        for i in range(collector_env_num)
    }
    batch_size = len(data)  # 32
    # batch_size = int(len(episodic_reward) / len(data[0]['reward']))
    timesteps = len(data[0]['reward'])  # 42
    device = data[0]['reward'][0].device
    intrinsic_reward_type = 'add'
    intrisic_reward = episodic_reward * torch.clamp(inter_episodic_reward, min=1, max=5)
    tb_logger.add_scalar('intrinsic_reward/intrinsic_reward_max', intrisic_reward.max(), estimate_cnt)
    tb_logger.add_scalar('intrinsic_reward/intrinsic_reward_mean', intrisic_reward.mean(), estimate_cnt)
    tb_logger.add_scalar('intrinsic_reward/intrinsic_reward_min', intrisic_reward.min(), estimate_cnt)

    if not isinstance(data[0], (list, dict)):
        intrisic_reward = intrisic_reward.to(device)
        intrisic_reward = torch.chunk(intrisic_reward, intrisic_reward.shape[0], dim=0)
        for item, rew in zip(data, intrisic_reward):
            if intrinsic_reward_type == 'add':
                item['reward'] += rew * index_to_beta[data['beta']]
    else:  # rnn nstep
        intrisic_reward = intrisic_reward.to(device)
        intrisic_reward = torch.chunk(intrisic_reward, int(intrisic_reward.shape[0]), dim=0)  # tensor to tuple
        # reward = torch.chunk(reward, int(reward.shape[0]/len(data[0]['reward'])), dim=0)

        for i in range(batch_size):  # if nstep 64 batch_size
            for j in range(timesteps):  # 24 24=20+2*2 eps_len
                if j < timesteps - nstep:
                    bonus = torch.cat([intrisic_reward[i * timesteps + j + k] for k in range(nstep)], dim=0)
                    if intrinsic_reward_type == 'add':
                        data[i]['reward'][j] += bonus * index_to_beta[int(data[i]['beta'][j])]

        # for i in range(batch_size): #64 batch_size
        #     for j in range(eps_len): #24 24=20+2*2 eps_len
        #             if self.intrinsic_reward_type == 'add':
        #                 data[i]['reward'][j]+=reward[i*eps_len+j]
        #             elif self.intrinsic_reward_type == 'new':
        #                 data[i]['reward'][j]+=reward[i*eps_len+j]
        #             elif self.intrinsic_reward_type == 'assign':
        #                 data[i]['reward'][j]=reward[i*eps_len+j]
    return data, estimate_cnt

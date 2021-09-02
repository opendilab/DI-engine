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
from typing import Union, Optional, Dict, Callable, List
import numpy as np

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
        obs = item['obs'] # episode
        action = item['action']
        obs_list.append(obs)
        action_list.append(action) 
    return  obs_list, action_list

class InverseNetwork(nn.Module):
    def __init__(self, obs_shape: Union[int, SequenceType],action_shape, hidden_size_list: SequenceType) -> None:
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
            nn.Linear(hidden_size_list[-1] * 2, 512),
            nn.ReLU(),
            nn.Linear(512, action_shape)
        )
        # for param in self.target.parameters():
        #     param.requires_grad = False

    def forward(self, inputs: Dict, inference: bool = False) -> Dict:
        if inference:
            with torch.no_grad():
                cur_obs_embedding = self.embedding_net( inputs['obs'])
            return cur_obs_embedding
        else:
            # obs: torch.Tensor,next_obs: torch.Tensor
            cur_obs_embedding =  self.embedding_net(inputs['obs'])
            next_obs_embedding =  self.embedding_net(inputs['next_obs'])
            # get pred action
            obs_plus_next_obs = torch.cat([cur_obs_embedding, next_obs_embedding], dim=-1)
            pred_action_logits = self.inverse_net(obs_plus_next_obs)
            pred_action_probs = nn.Softmax(dim=-1)(pred_action_logits)
        return pred_action_probs


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
        self.episodic_reward_model = InverseNetwork(config.obs_shape,config.action_shape, config.hidden_size_list)
        self.episodic_reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        # self.train_data = []
        self.train_obs = []
        self.train_next_obs = []
        self.train_action = []
        self.opt = optim.Adam(self.episodic_reward_model.parameters(), config.learning_rate)
        self.estimate_cnt = 0

    def _train(self) -> None:
        # stack episode dim
        self.train_obs = [torch.stack(episode_obs[:-1],dim=0) for episode_obs in self.train_obs] # self.train_obs list(list) 32,42 batch_size,timesteps,dim
        self.train_next_obs = [torch.stack(episode_obs[1:],dim=0) for episode_obs in self.train_obs]
        self.train_action = [torch.stack( episode_action[:-1],dim=0) for episode_action in self.train_action]
        
        # stack batch dim
        if len(self.cfg.obs_shape) == 3:
            self.train_obs = torch.stack(self.train_obs,dim=0).view(self.train_obs.shape[0]*self.train_obs.shape[1],*self.cfg.obs_shape) # -1) TODO image
        else:
            self.train_obs = torch.stack(self.train_obs,dim=0).view(self.train_obs.shape[0]*self.train_obs.shape[1],self.cfg.obs_shape) # -1) TODO image
        self.train_next_obs = torch.stack(self.train_next_obs,dim=0).view(self.train_next_obs.shape[0]*self.train_next_obs.shape[1],-1)
        self.train_action = torch.stack(self.train_action,dim=0).view(self.train_action.shape[0]*self.train_action.shape[1],-1)
        # train_obs: list = random.sample(self.train_obs, self.cfg.batch_size)
        # train_obs: list = random.sample(self.train_data, self.cfg.batch_size)
        # sample episode's timestep index
        train_index = np.random.randint(low=0,high= self.train_obs.shape[0]*self.train_obs.shape[1],size=64) # self.cfg.reward_model_batch_size) 

        train_obs: torch.Tensor = self.train_obs[train_index].to(self.device) # shape reward_model_batch_size, obs_dim
        train_next_obs: torch.Tensor = self.train_next_obs[train_index].to(self.device)
        train_action: torch.Tensor = self.train_action[train_index].to(self.device)

        train_data = {'obs': train_obs ,'next_obs': train_next_obs }
        pred_action_probs = self.episodic_reward_model(train_data)

        inverse_loss = nn.CrossEntropyLoss(pred_action_probs, train_action).mean(dim=1)
        self.opt.zero_grad()
        inverse_loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect): # self.cfg.update_per_collect_intrinsic_reward
            self._train()
        self.clear_data()

    def _compute_intrinsic_reward(self, episodic_memory: List,
        current_c_state: torch.Tensor,k=10,
        kernel_cluster_distance=0.008,
        kernel_epsilon=0.0001,
        c=0.001,
        sm=8,) -> float:
        state_dist = [(c_state, torch.dist(c_state, current_c_state)) for c_state in episodic_memory]
        state_dist.sort(key=lambda x: x[1])
        state_dist = state_dist[:k]
        dist = [d[1].item() for d in state_dist]
        dist = np.array(dist)

        # TODO: moving average
        dist = dist / np.mean(dist)

        dist = np.max(dist - kernel_cluster_distance, 0)
        kernel = kernel_epsilon / (dist + kernel_epsilon)
        s = np.sqrt(np.sum(kernel)) + c

        if np.isnan(s) or s > sm:
            return 0
        return 1 / s

    def estimate(self, data: list) -> None:
        """
        Rewrite the reward key in each row of the data.
        """
        obs = collect_states(data)
        if isinstance(obs[0], list):  # if self.train_data list( list(torch.tensor) )
            tmp = []
            for i in range(len(data)):
                tmp += obs[i]
            obs = tmp
        obs = torch.stack(obs).to(self.device)
        
        with torch.no_grad():
            cur_obs_embedding = self.episodic_reward_model(obs,inference=True)
            episodic_reward = [[]*cur_obs_embedding.shape[0]]
            for i in range(cur_obs_embedding.shape[0]):
                episodic_memory = list(cur_obs_embedding[i])
                for j in range(cur_obs_embedding.shape[1]):
                    reward = self._compute_intrinsic_reward(episodic_memory,cur_obs_embedding[i][j])
                    episodic_reward[i].append(reward)
        
            self.estimate_cnt += 1
            self.tb_logger.add_scalar('intrinsic_reward_max', episodic_reward.max(), self.estimate_cnt)
            self.tb_logger.add_scalar('intrinsic_reward_min', episodic_reward.min(), self.estimate_cnt)
            episodic_reward = (episodic_reward - episodic_reward.min()) / (episodic_reward.max() - episodic_reward.min() + 1e-8)
        
        return episodic_reward
           

    def collect_data(self, data: list) -> None:
        train_obs, train_action = collect_states_episodic(data)
        self.train_obs.extend(train_obs)
        self.train_action.extend(train_action)

    def clear_data(self) -> None:
        self.train_obs.clear()
        self.train_action.clear()


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
        self.estimate_cnt = 0

    def _train(self) -> None:
        if isinstance(self.train_data[0], list):  # if self.train_data list( list(torch.tensor) ) rnn
            tmp = []
            for i in range(len(self.train_data)):
                tmp += self.train_data[i]
            self.train_data = tmp
        train_data: list = random.sample(self.train_data, self.cfg.batch_size)
        train_data: torch.Tensor = torch.stack(train_data).to(self.device)
        predict_feature, target_feature = self.reward_model(train_data)
        loss = F.mse_loss(predict_feature, target_feature.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            self._train()
        self.clear_data()

    def estimate(self, data: list) -> None:
        """
        Rewrite the reward key in each row of the data.
        """
        obs = collect_states(data)
        if isinstance(obs[0], list):  # if self.train_data list( list(torch.tensor) )
            tmp = []
            for i in range(len(data)):
                tmp += obs[i]
            obs = tmp
        obs = torch.stack(obs).to(self.device)
        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(obs)
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            self.estimate_cnt += 1
            self.tb_logger.add_scalar('intrinsic_reward_max', reward.max(), self.estimate_cnt)
            self.tb_logger.add_scalar('intrinsic_reward_min', reward.min(), self.estimate_cnt)
            reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-8)
        return reward
            # if not isinstance(data[0], (list, dict)):
            #     reward = reward.to(data[0]['reward'].device)
            #     reward = torch.chunk(reward, reward.shape[0], dim=0)
            #     for item, rew in zip(data, reward):
            #         if self.intrinsic_reward_type == 'add':
            #             item['reward'] += rew
            #         elif self.intrinsic_reward_type == 'new':
            #             item['intrinsic_reward'] = rew
            #         elif self.intrinsic_reward_type == 'assign':
            #             item['reward'] = rew
            # else:  #rnn nstep
            #     reward = reward.to(data[0]['reward'][0].device)
            #     reward = torch.chunk(reward, int(reward.shape[0]), dim=0)
            #     # reward = torch.chunk(reward, int(reward.shape[0]/len(data[0]['reward'])), dim=0)
            #     # reward.shape[0] 64 batch_size len(data[0]['reward'])) 24=20+2*2 eps_len
            #     batch_size = int(reward.__len__() / len(data[0]['reward']))
            #     eps_len = len(data[0]['reward'])  ## pu if r2d2 n-step reward   fake_reward = torch.zeros(1)
            #     for i in range(batch_size):  #64 batch_size
            #         for j in range(eps_len):  #24 24=20+2*2 eps_len
            #             if j < eps_len - self.cfg.nstep:
            #                 bonus = torch.cat([reward[i * eps_len + j + k] for k in range(self.cfg.nstep)], dim=0)
            #                 if self.intrinsic_reward_type == 'add':
            #                     data[i]['reward'][j] += bonus
            #                 # elif self.intrinsic_reward_type == 'new':
            #                 #     data[i]['reward'][j]+= bonus
            #                 elif self.intrinsic_reward_type == 'assign':
            #                     data[i]['reward'][j] = bonus

                # for i in range(batch_size): #64 batch_size
                #     for j in range(eps_len): #24 24=20+2*2 eps_len
                #             if self.intrinsic_reward_type == 'add':
                #                 data[i]['reward'][j]+=reward[i*eps_len+j]
                #             elif self.intrinsic_reward_type == 'new':
                #                 data[i]['reward'][j]+=reward[i*eps_len+j]
                #             elif self.intrinsic_reward_type == 'assign':
                #                 data[i]['reward'][j]=reward[i*eps_len+j]

    def collect_data(self, data: list) -> None:
        self.train_data.extend(collect_states(data))

    def clear_data(self) -> None:
        self.train_data.clear()


def  fusion_reward(data, episodic_reward,inter_episodic_reward,nstep,collector_env_num):
    # index_to_eps = {i: 0.4 ** (1 + 8 * i / (self._env_num - 1)) for i in range(self._env_num)}
    index_to_beta = {
        i: 0.3 * torch.sigmoid(torch.tensor(10 * (2 * i - (collector_env_num- 2)) / (collector_env_num - 2)))
        for i in range(collector_env_num)
    }
    index_to_gamma = {
        i: 1 - torch.exp(
            ((collector_env_num - 1 - i) * torch.log(torch.tensor(1 - 0.997)) + i * torch.log(torch.tensor(1 - 0.99))) /
            (collector_env_num - 1)
    )
    for i in range(collector_env_num)
    }
    intrinsic_reward_type = 'add'
    intrisic_reward = episodic_reward * torch.min(torch.max(inter_episodic_reward,1),5)
    if not isinstance(data[0], (list, dict)):
            intrisic_reward = intrisic_reward.to(data[0]['reward'].device)
            intrisic_reward = torch.chunk(intrisic_reward, intrisic_reward.shape[0], dim=0)
            for item, rew in zip(data, intrisic_reward):
                if intrinsic_reward_type == 'add':
                    item['reward'] += rew * index_to_beta[data['beta']]
    else:  #rnn nstep
        intrisic_reward = intrisic_reward.to(data[0]['reward'][0].device)
        intrisic_reward = torch.chunk(intrisic_reward, int(intrisic_reward.shape[0]), dim=0)
        # reward = torch.chunk(reward, int(reward.shape[0]/len(data[0]['reward'])), dim=0)
        # reward.shape[0] 64 batch_size len(data[0]['reward'])) 24=20+2*2 eps_len
        batch_size = int(intrisic_reward.__len__() / len(data[0]['reward']))
        eps_len = len(data[0]['reward'])  ## pu if r2d2 n-step reward   fake_reward = torch.zeros(1)
        for i in range(batch_size):  # if nstep 64 batch_size
            for j in range(eps_len):  #24 24=20+2*2 eps_len
                if j < eps_len - nstep:
                    bonus = torch.cat([intrisic_reward[i * eps_len + j + k] for k in range(nstep)], dim=0)
                    if intrinsic_reward_type == 'add':
                        data[i]['reward'][j] += bonus *  index_to_beta[data['beta']]
                    
        # for i in range(batch_size): #64 batch_size
        #     for j in range(eps_len): #24 24=20+2*2 eps_len
        #             if self.intrinsic_reward_type == 'add':
        #                 data[i]['reward'][j]+=reward[i*eps_len+j]
        #             elif self.intrinsic_reward_type == 'new':
        #                 data[i]['reward'][j]+=reward[i*eps_len+j]
        #             elif self.intrinsic_reward_type == 'assign':
        #                 data[i]['reward'][j]=reward[i*eps_len+j]
    return data
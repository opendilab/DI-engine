from typing import List, Dict, Any, Tuple, Union, Optional
from easydict import EasyDict

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Independent, Normal
import copy

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.model import FCEncoder, ConvEncoder
from .base_reward_model import BaseRewardModel


class GuidedCostNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128,
        output_size=1,
    ):
        super(GuidedCostNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


@REWARD_MODEL_REGISTRY.register('guided_cost')
class GuidedCostRewardModel(BaseRewardModel):
    r"""
    Overview:
        Policy class of Guided cost algorithm.

        https://arxiv.org/pdf/1603.00448.pdf

    """

    config = dict(
        type='guided_cost',
        learning_rate=1e-3,
        action_shape=1,
        continuous=True,
        batch_size=64,
        hidden_size=128,
        update_per_collect=100,
        log_every_n_train=50,
        store_model_every_n_train=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(GuidedCostRewardModel, self).__init__()
        self.cfg = config
        self.action_shape = self.cfg.action_shape
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = GuidedCostNN(config.input_size, config.hidden_size)
        self.reward_model.to(self.device)
        self.opt = optim.Adam(self.reward_model.parameters(), lr=config.learning_rate)

    def train(self, expert_demo: torch.Tensor, samp: torch.Tensor, iter, step):
        device_0 = expert_demo[0]['obs'].device
        device_1 = samp[0]['obs'].device
        for i in range(len(expert_demo)):
            expert_demo[i]['prob'] = torch.FloatTensor([1]).to(device_0)
        if self.cfg.continuous:
            for i in range(len(samp)):
                (mu, sigma) = samp[i]['logit']
                dist = Independent(Normal(mu, sigma), 1)
                next_action = samp[i]['action']
                log_prob = dist.log_prob(next_action)
                samp[i]['prob'] = torch.exp(log_prob).unsqueeze(0).to(device_1)
        else:
            for i in range(len(samp)):
                probs = F.softmax(samp[i]['logit'], dim=-1)
                prob = probs[samp[i]['action']]
                samp[i]['prob'] = prob.to(device_1)
        # Mix the expert data and sample data to train the reward model.
        samp.extend(expert_demo)
        expert_demo = default_collate(expert_demo)
        samp = default_collate(samp)
        cost_demo = self.reward_model(
            torch.cat([expert_demo['obs'], expert_demo['action'].float().reshape(-1, self.action_shape)], dim=-1)
        )
        cost_samp = self.reward_model(
            torch.cat([samp['obs'], samp['action'].float().reshape(-1, self.action_shape)], dim=-1)
        )

        prob = samp['prob'].unsqueeze(-1)
        loss_IOC = torch.mean(cost_demo) + \
            torch.log(torch.mean(torch.exp(-cost_samp)/(prob+1e-7)))
        # UPDATING THE COST FUNCTION
        self.opt.zero_grad()
        loss_IOC.backward()
        self.opt.step()
        if iter % self.cfg.log_every_n_train == 0:
            self.tb_logger.add_scalar('reward_model/loss_iter', loss_IOC, iter)
            self.tb_logger.add_scalar('reward_model/loss_step', loss_IOC, step)

    def estimate(self, data: list) -> List[Dict]:
        # NOTE: this estimate method of gcl alg. is a little different from the one in other irl alg.,
        # because its deepcopy is operated before learner train loop.
        train_data_augmented = data
        for i in range(len(train_data_augmented)):
            with torch.no_grad():
                reward = self.reward_model(
                    torch.cat([train_data_augmented[i]['obs'], train_data_augmented[i]['action'].float()]).unsqueeze(0)
                ).squeeze(0)
                train_data_augmented[i]['reward'] = -reward

        return train_data_augmented

    def collect_data(self, data) -> None:
        """
        Overview:
            Collecting training data, not implemented if reward model (i.e. online_net) is only trained ones, \
                if online_net is trained continuously, there should be some implementations in collect_data method
        """
        # if online_net is trained continuously, there should be some implementations in collect_data method
        pass

    def clear_data(self):
        """
        Overview:
            Collecting clearing data, not implemented if reward model (i.e. online_net) is only trained ones, \
                if online_net is trained continuously, there should be some implementations in clear_data method
        """
        # if online_net is trained continuously, there should be some implementations in clear_data method
        pass

    def state_dict_reward_model(self) -> Dict[str, Any]:
        return {
            'model': self.reward_model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }

    def load_state_dict_reward_model(self, state_dict: Dict[str, Any]) -> None:
        self.reward_model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['optimizer'])

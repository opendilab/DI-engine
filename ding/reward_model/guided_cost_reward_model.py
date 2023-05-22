from typing import List, Dict, Any
from easydict import EasyDict
from ditk import logging

import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from ding.utils import REWARD_MODEL_REGISTRY
from ding.utils.data import default_collate
from .base_reward_model import BaseRewardModel
from .network import GCLNetwork


@REWARD_MODEL_REGISTRY.register('guided_cost')
class GuidedCostRewardModel(BaseRewardModel):
    """
    Overview:
        Policy class of Guided cost algorithm. (https://arxiv.org/pdf/1603.00448.pdf)
    Interface:
        ``estimate``, ``train``, ``collect_data``, ``clear_date``, \
        ``__init__``,  ``state_dict``, ``load_state_dict``, ``learn``\
        ``state_dict_reward_model``, ``load_state_dict_reward_model``
    Config:
        == ====================  ========   =============  ========================================  ================
        ID Symbol                Type       Default Value  Description                               Other(Shape)
        == ====================  ========   =============  ========================================  ================
        1  ``type``              str         guided_cost   | Reward model register name, refer        |
                                                           | to registry ``REWARD_MODEL_REGISTRY``    |
        2  | ``continuous``      bool        True          | Whether action is continuous             |
        3  | ``learning_rate``   float       0.001         | learning rate for optimizer              |
        4  | ``update_per_``     int         100           | Number of updates per collect            |
           | ``collect``                                   |                                          |
        5  | ``batch_size``      int         64            | Training batch size                      |
        6  | ``hidden_size``     int         128           | Linear model hidden size                 |
        7  | ``action_shape``    int         1             | Action space shape                       |
        8  | ``log_every_n``     int         50            | add loss to log every n iteration        |
           | ``_train``                                    |                                          |
        9  | ``store_model_``    int         100           | save model every n iteration             |
           | ``every_n_train``                                                                        |
        == ====================  ========   =============  ========================================  ================

    """

    config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='guided_cost',
        # (float) The step size of gradient descent.
        learning_rate=1e-3,
        # (int) Action space shape, such as 1.
        action_shape=1,
        # (bool) Whether action is continuous.
        continuous=True,
        # (int) How many samples in a training batch.
        batch_size=64,
        # (int) Linear model hidden size.
        hidden_size=128,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        update_per_collect=100,
        # (int) Add loss to log every n iteration.
        log_every_n_train=50,
        # (int) Save model every n iteration.
        store_model_every_n_train=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(GuidedCostRewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.iter = 0
        self.reward_model = GCLNetwork(
            config.input_size, [config.hidden_size, config.hidden_size],
            output_size=1,
            action_shape=config.action_shape
        )
        self.reward_model.to(self.device)
        self.opt = optim.Adam(self.reward_model.parameters(), lr=config.learning_rate)
        self.train_data = []
        self.load_expert_data()

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config['expert_data_path']`` attribute in self.
        Effects:
            This is a side effect function which updates the expert data attribute (e.g.  ``self.expert_data``)
        """
        with open(self.cfg.expert_data_path, 'rb') as f:
            self.expert_data = pickle.load(f)

    def train(self) -> None:
        """
        Overview:
            Train the reward model.
        """
        # sample data for expert and train data
        sample_size = min(len(self.expert_data), self.cfg.batch_size)
        expert_demo = random.sample(self.expert_data, sample_size)
        samp = random.sample(self.train_data, sample_size)

        # remove non-tensor data in data list
        samp = self._remove_redundant_keys(samp)

        # train the reward model
        for _ in range(self.cfg.update_per_collect):
            loss_ioc = self._train(expert_demo, samp)
            self.tb_logger.add_scalar('reward_model/loss_iter', loss_ioc, self.iter)
            self.iter += 1

    def _train(self, expert_demo: torch.Tensor, samp: torch.Tensor) -> float:
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
        loss_IOC = self.reward_model.learn(expert_demo, samp)
        # UPDATING THE COST FUNCTION
        self.opt.zero_grad()
        loss_IOC.backward()
        self.opt.step()

        return loss_IOC.item()

    def estimate(self, data: list) -> List[Dict]:
        # NOTE: this estimate method of gcl alg. is a little different from the one in other irl alg.,
        # because its deepcopy is operated before learner train loop.
        train_data_augmented = data
        for i in range(len(train_data_augmented)):
            with torch.no_grad():
                reward = self.reward_model.forward(
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
        self.train_data.extend(data)

    def clear_data(self, iter: int):
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

    def _remove_redundant_keys(self, samp: List[Dict]) -> List[Dict]:
        """
        Overview:
            Remove redundant keys in the data list.
        Arguments:
            - samp (:obj:`List[Dict]`): The data list.
        Returns:
            - (:obj:`List[Dict]`): The data list without redundant keys.
        """
        keeped_keys = ['obs', 'next_obs', 'action', 'logit']
        assert samp is not None and bool(samp), "samp is empty."
        assert all(key in samp[0] for key in keeped_keys), "samp is missing required keys."
        fixed_samp = []
        for item in samp:
            fixed_item = {}
            for key in keeped_keys:
                fixed_item[key] = item[key]
            fixed_samp.append(fixed_item)
        return fixed_samp

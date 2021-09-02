from typing import Any, List
from easydict import EasyDict
import numpy as np
import torch
import pickle
import scipy.stats as stats
from torch.functional import Tensor
from ding.utils import REWARD_MODEL_REGISTRY, one_time_warning
from .base_reward_model import BaseRewardModel
from ding.model.common.autoencoder import AutoEncoder
from collections import deque
import random
from ding.config.config import deep_merge_dicts
import logging
EPS = 1e-8


@REWARD_MODEL_REGISTRY.register('countbased')
class CountBasedRewardModel(BaseRewardModel):
    """
    Overview:
        The Count-Based reward model class, more detail you can find: https://arxiv.org/abs/1611.04717
    Interface:
        ``estimate``, ``train``, ``collect_data``, ``clear_data``, \
            ``__init__``, ``_train``
    """
    config = dict(
        type='countbased',
        counter_type='SimHash',  # support countertype: SimHash, AutoEncoder
        bonus_coefficent=0.5,
        state_dim=2,
        hash_dim=32,
        # below features are only for AutoEncoder counter type
        max_buff_len=10000,
        batch_size=4,
        update_per_iter=3,
        learning_rate=0.1,
    )

    def __init__(self, cfg: dict, device, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`Dict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`str`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(CountBasedRewardModel, self).__init__()
        self.cfg: EasyDict = EasyDict(deep_merge_dicts(self.config, cfg))
        self._beta = self.cfg.bonus_coefficent
        self._counter_type = self.cfg.counter_type
        self.device = device
        self.tb_logger = tb_logger
        assert self._counter_type in ['SimHash', 'AutoEncoder']
        print(self.cfg)
        if self._counter_type == 'SimHash':
            self._counter = SimHash(self.cfg.state_dim, self.cfg.hash_dim, self.device)
            self.estimate_iter = 0
        elif self._counter_type == 'AutoEncoder':
            self._counter = AutoEncoderCounter(
                self.cfg.state_dim, self.cfg.hash_dim, self.device, learning_rate=self.cfg.learning_rate
            )
            self.replay_buf = deque(maxlen=self.cfg.max_buff_len)
            self.batch_size = self.cfg.batch_size
            self.update_per_iter = self.cfg.update_per_iter
            self.train_iter = 0

    def train(self) -> None:
        """
        Overview:
            Training the autoencoder model.
        """
        if self._counter_type == 'AutoEncoder':
            if len(self.replay_buf) > self.batch_size:
                data = random.sample(self.replay_buf, self.batch_size)
                data = torch.stack(data, dim=0).to(self.device)
                total_loss = self._counter.train(data)
                self.tb_logger.add_scalar('reward_model/autoencoder_loss', total_loss, self.train_iter)
                self.train_iter += 1
            else:
                logging.warning(f"no enough data {len(self.replay_buf)}/{self.batch_size}")

    def estimate(self, data: list) -> None:
        """
        Overview:
            Estimate reward by rewriting the reward keys.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation,\
                 with at least ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        if self._counter_type == 'SimHash':
            s = torch.stack([item['obs'] for item in data], dim=0).to(self.device)
            hash_cnts = self._counter.update(s)
            self.tb_logger.add_scalar('reward_model/statehash_size', len(self._counter.hash), self.estimate_iter)
            for item, cnt in zip(data, hash_cnts):
                item['reward'] = item['reward'] + self._beta / np.sqrt(cnt)
            self.estimate_iter += 1
        else:
            self.collect_data(data)
            if self.train_iter % self.update_per_iter == 0:
                self.train()
            # like origin paper, only use the latest frame image as hash target
            s = torch.stack([item['obs'][-1:] for item in data], dim=0).to(self.device)
            hash_cnts = self._counter.update(s)
            self.tb_logger.add_scalar('reward_model/statehash_size', len(self._counter.hash), self.train_iter)
            self.tb_logger.add_scalar('reward_model/buffer_size', len(self.replay_buf), self.train_iter)
            for item, cnt in zip(data, hash_cnts):
                item['reward'] = item['reward'] + self._beta / np.sqrt(cnt)

    def collect_data(self, data: list):
        """
        Overview:
            Collecting training data by iterating data items in the input list
        Arguments:
            - data (:obj:`list`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self`` by \
                iterating data items in the input data items' list
        """
        if self._counter_type == 'AutoEncoder':
            self.replay_buf.extend([item['obs'][-1:] for item in data])

    def clear_data(self):
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        pass


class SimHash():

    def __init__(self, state_emb: int, k: int, device) -> None:
        self.hash = {}
        self.fc = torch.nn.Linear(state_emb, k, bias=False).to(device)
        # calcuating  hash value need a random matrix sampled from gaussion distribution
        self.fc.weight.data = torch.randn_like(self.fc.weight)
        self.device = device

    def update(self, states: torch.Tensor) -> list:
        counts = []
        if len(states.size()) > 2:
            states = torch.flatten(states, 1)
        # hash value = sign(A*states)
        keys = np.sign(self.fc(states).to('cpu').detach().numpy()).tolist()
        for idx, key in enumerate(keys):
            key = str(key)
            keys[idx] = key
            self.hash[key] = self.hash.get(key, 0) + 1
        for key in keys:
            counts.append(self.hash[key])
        return counts


class AutoEncoderCounter():

    def __init__(self, state_dim: list, k: int, device, learning_rate: float = 1e-3, mu: float = 1.0) -> None:
        self.hash = {}
        self.autoencoder = AutoEncoder(state_dim=state_dim).to(device)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)
        self.fc = torch.nn.Linear(256, k, bias=False).to(device)
        # this part is similar as SimHash
        self.fc.weight.data = torch.randn_like(self.fc.weight)
        self.device = device
        self.pixel_value = 64
        self.mu = mu

    def update(self, states: torch.Tensor) -> list:
        counts = []
        self.autoencoder.eval()
        with torch.no_grad():
            states_emb = self.autoencoder.generate(states)
            states_emb = torch.round(states_emb)
            keys = np.sign(self.fc(states_emb).to('cpu').detach().numpy()).tolist()
        for idx, key in enumerate(keys):
            key = str(key)
            keys[idx] = key
            self.hash[key] = self.hash.get(key, 0) + 1
        for key in keys:
            counts.append(self.hash[key])
        return counts

    def train(self, states: torch.Tensor, noise_intensity: float = 0.3) -> Any:
        self.autoencoder.train()
        # generate embedding in latent space
        state_emb = self.autoencoder.generate(states)
        # add noise to embedding. max intensity * [-1,1]
        state_emb_noise = state_emb+noise_intensity * \
            (2*torch.rand_like(state_emb)-1)
        # reconstruct from latent space
        rec_image = self.autoencoder.reconstruct(state_emb_noise)
        origin_image = torch.round(states * (self.pixel_value - 1)).long()
        # reconstruct loss
        rec_loss = -(rec_image.gather(1, origin_image) + EPS).log()
        # let embedding more close to 0 or 1
        embedding_loss = torch.min((1 - state_emb).pow(2), state_emb.pow(2))
        total_loss = rec_loss.sum([1, 2, 3]).mean() + \
            self.mu*embedding_loss.mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return total_loss.item()

from typing import Union, Tuple, List, Dict
from easydict import EasyDict

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from .base_reward_model import BaseRewardModel
from ding.utils import RunningMeanStd
from ding.torch_utils.data_helper import to_tensor
import numpy as np


def collect_states(iterator):
    res = []
    for item in iterator:
        state = item['obs']
        res.append(state)
    return res


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
    """
    Overview:
        The RND reward model class (https://arxiv.org/abs/1810.12894v1)
    Interface:
        ``estimate``, ``train``, ``collect_data``, ``clear_data``, \
            ``__init__``, ``_train``, ``load_state_dict``, ``state_dict``
    Config:
        == ====================  =====  =============  =======================================  =======================
        ID Symbol                Type   Default Value  Description                              Other(Shape)
        == ====================  =====  =============  =======================================  =======================
        1   ``type``              str     rnd          | Reward model register name, refer      |
                                                       | to registry ``REWARD_MODEL_REGISTRY``  |
        2  | ``intrinsic_``      str      add          | the intrinsic reward type              | including add, new
           | ``reward_type``                           |                                        | , or assign
        3  | ``learning_rate``   float    0.001        | The step size of gradient descent      |
        4  | ``batch_size``      int      64           | Training batch size                    |
        5  | ``hidden``          list     [64, 64,     | the MLP layer shape                    |
           | ``_size_list``      (int)    128]         |                                        |
        6  | ``update_per_``     int      100          | Number of updates per collect          |
           | ``collect``                               |                                        |
        7  | ``obs_norm``        bool     True         | Observation normalization              |
        8  | ``obs_norm_``       int      0            | min clip value for obs normalization   |
           | ``clamp_min``
        9  | ``obs_norm_``       int      1            | max clip value for obs normalization   |
           | ``clamp_max``
        10 | ``intrinsic_``      float    0.01         | the weight of intrinsic reward         | r = w*r_i + r_e
             ``reward_weight``
        11 | ``extrinsic_``      bool     True         | Whether to normlize extrinsic reward
             ``reward_norm``
        12 | ``extrinsic_``      int      1            | the upper bound of the reward
            ``reward_norm_max``                        | normalization
        == ====================  =====  =============  =======================================  =======================
    """
    config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='rnd',
        # (str) The intrinsic reward type, including add, new, or assign.
        intrinsic_reward_type='add',
        # (float) The step size of gradient descent.
        learning_rate=1e-3,
        # (float) Batch size.
        batch_size=64,
        # (list(int)) Sequence of ``hidden_size`` of reward network.
        # If obs.shape == 1,  use MLP layers.
        # If obs.shape == 3,  use conv layer and final dense layer.
        hidden_size_list=[64, 64, 128],
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        update_per_collect=100,
        # (bool) Observation normalization: transform obs to mean 0, std 1.
        obs_norm=True,
        # (int) Min clip value for observation normalization.
        obs_norm_clamp_min=-1,
        # (int) Max clip value for observation normalization.
        obs_norm_clamp_max=1,
        # Means the relative weight of RND intrinsic_reward.
        # (float) The weight of intrinsic reward
        # r = intrinsic_reward_weight * r_i + r_e.
        intrinsic_reward_weight=0.01,
        # (bool) Whether to normlize extrinsic reward.
        # Normalize the reward to [0, extrinsic_reward_norm_max].
        extrinsic_reward_norm=True,
        # (int) The upper bound of the reward normalization.
        extrinsic_reward_norm_max=1,
    )

    def __init__(self, config: EasyDict, device: str = 'cpu', tb_logger: 'SummaryWriter' = None) -> None:  # noqa
        super(RndRewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        if tb_logger is None:  # TODO
            from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter('rnd_reward_model')
        self.tb_logger = tb_logger
        self.reward_model = RndNetwork(config.obs_shape, config.hidden_size_list)
        self.reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_obs = []
        self.opt = optim.Adam(self.reward_model.predictor.parameters(), config.learning_rate)
        self._running_mean_std_rnd_reward = RunningMeanStd(epsilon=1e-4)
        self.estimate_cnt_rnd = 0
        self.train_cnt_icm = 0
        self._running_mean_std_rnd_obs = RunningMeanStd(epsilon=1e-4)

    def _train(self) -> None:
        train_data: list = random.sample(self.train_obs, self.cfg.batch_size)
        train_data: torch.Tensor = torch.stack(train_data).to(self.device)
        if self.cfg.obs_norm:
            # Note: observation normalization: transform obs to mean 0, std 1
            self._running_mean_std_rnd_obs.update(train_data.cpu().numpy())
            train_data = (train_data - to_tensor(self._running_mean_std_rnd_obs.mean).to(self.device)) / to_tensor(
                self._running_mean_std_rnd_obs.std
            ).to(self.device)
            train_data = torch.clamp(train_data, min=self.cfg.obs_norm_clamp_min, max=self.cfg.obs_norm_clamp_max)

        predict_feature, target_feature = self.reward_model(train_data)
        loss = F.mse_loss(predict_feature, target_feature.detach())
        self.tb_logger.add_scalar('rnd_reward/loss', loss, self.train_cnt_icm)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            self._train()
            self.train_cnt_icm += 1

    def estimate(self, data: list) -> List[Dict]:
        """
        Rewrite the reward key in each row of the data.
        """
        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        train_data_augmented = self.reward_deepcopy(data)

        obs = collect_states(train_data_augmented)
        obs = torch.stack(obs).to(self.device)
        if self.cfg.obs_norm:
            # Note: observation normalization: transform obs to mean 0, std 1
            obs = (obs - to_tensor(self._running_mean_std_rnd_obs.mean
                                   ).to(self.device)) / to_tensor(self._running_mean_std_rnd_obs.std).to(self.device)
            obs = torch.clamp(obs, min=self.cfg.obs_norm_clamp_min, max=self.cfg.obs_norm_clamp_max)

        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(obs)
            mse = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            self._running_mean_std_rnd_reward.update(mse.cpu().numpy())

            # Note: according to the min-max normalization, transform rnd reward to [0,1]
            rnd_reward = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)

            # save the rnd_reward statistics into tb_logger
            self.estimate_cnt_rnd += 1
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_max', rnd_reward.max(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_mean', rnd_reward.mean(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_min', rnd_reward.min(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_std', rnd_reward.std(), self.estimate_cnt_rnd)

            rnd_reward = rnd_reward.to(self.device)
            rnd_reward = torch.chunk(rnd_reward, rnd_reward.shape[0], dim=0)
        """
        NOTE: Following normalization approach to extrinsic reward seems be not reasonable,
        because this approach compresses the extrinsic reward magnitude, resulting in less informative reward signals.
        """
        # rewards = torch.stack([data[i]['reward'] for i in range(len(data))])
        # rewards = (rewards - torch.min(rewards)) / (torch.max(rewards) - torch.min(rewards))

        for item, rnd_rew in zip(train_data_augmented, rnd_reward):
            if self.intrinsic_reward_type == 'add':
                if self.cfg.extrinsic_reward_norm:
                    item['reward'] = item[
                        'reward'] / self.cfg.extrinsic_reward_norm_max + rnd_rew * self.cfg.intrinsic_reward_weight
                else:
                    item['reward'] = item['reward'] + rnd_rew * self.cfg.intrinsic_reward_weight
            elif self.intrinsic_reward_type == 'new':
                item['intrinsic_reward'] = rnd_rew
                if self.cfg.extrinsic_reward_norm:
                    item['reward'] = item['reward'] / self.cfg.extrinsic_reward_norm_max
            elif self.intrinsic_reward_type == 'assign':
                item['reward'] = rnd_rew

        # save the augmented_reward statistics into tb_logger
        rew = [item['reward'].cpu().numpy() for item in train_data_augmented]
        self.tb_logger.add_scalar('augmented_reward/reward_max', np.max(rew), self.estimate_cnt_rnd)
        self.tb_logger.add_scalar('augmented_reward/reward_mean', np.mean(rew), self.estimate_cnt_rnd)
        self.tb_logger.add_scalar('augmented_reward/reward_min', np.min(rew), self.estimate_cnt_rnd)
        self.tb_logger.add_scalar('augmented_reward/reward_std', np.std(rew), self.estimate_cnt_rnd)
        return train_data_augmented

    def collect_data(self, data: list) -> None:
        self.train_obs.extend(collect_states(data))

    def clear_data(self) -> None:
        self.train_obs.clear()

    def state_dict(self) -> Dict:
        return self.reward_model.state_dict()

    def load_state_dict(self, _state_dict: Dict) -> None:
        self.reward_model.load_state_dict(_state_dict)

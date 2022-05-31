from typing import Union, Tuple, List, Dict, Any
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
import copy


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
    config = dict(
        type='rnd',
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        batch_size=64,
        hidden_size_list=[64, 64, 128],
        update_per_collect=100,
        obs_norm=True,
        obs_norm_clamp_min=-1,
        obs_norm_clamp_max=1,
        intrinsic_reward_weight=None,
        # means the relative weight of RND intrinsic_reward.
        # If intrinsic_reward_weight=None, we will automatically set it based on
        # the absolute value of the difference between max and min extrinsic reward in the sampled mini-batch
        # please refer to  estimate() method for details.
        intrinsic_reward_rescale=0.01,
        # means the rescale value of RND intrinsic_reward only used when intrinsic_reward_weight is None
    )

    def __init__(self, config: EasyDict, device: str = 'cpu', tb_logger: 'SummaryWriter' = None) -> None:  # noqa
        super(RndRewardModel, self).__init__()
        self.cfg = config
        self.intrinsic_reward_rescale = self.cfg.intrinsic_reward_rescale
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
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            self._train()

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
            rnd_reward = (mse - mse.min()) / (mse.max() - mse.min() + 1e-11)

            self.estimate_cnt_rnd += 1
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_max', rnd_reward.max(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_mean', rnd_reward.mean(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_min', rnd_reward.min(), self.estimate_cnt_rnd)

            rnd_reward = rnd_reward.to(train_data_augmented[0]['reward'].device)
            rnd_reward = torch.chunk(rnd_reward, rnd_reward.shape[0], dim=0)
        """
        NOTE: Following normalization approach to extrinsic reward seems be not reasonable,
        because this approach compresses the extrinsic reward magnitude, resulting in less informative reward signals.
        """
        # rewards = torch.stack([data[i]['reward'] for i in range(len(data))])
        # rewards = (rewards - torch.min(rewards)) / (torch.max(rewards) - torch.min(rewards))

        # TODO(pu): how to set intrinsic_reward_rescale automatically?
        if self.cfg.intrinsic_reward_weight is None:
            """
            NOTE: the following way of setting self.cfg.intrinsic_reward_weight is only suitable for the dense
            reward env like lunarlander, not suitable for the dense reward env.
            In sparse reward env, e.g. minigrid, if the agent reaches the goal, it obtain reward ~1, otherwise 0.
            Thus, in sparse reward env, it's reasonable to set the intrinsic_reward_weight approximately equal to
             the inverse of max_episode_steps.
            """
            self.cfg.intrinsic_reward_weight = self.intrinsic_reward_rescale * max(
                1,
                abs(
                    max([train_data_augmented[i]['reward'] for i in range(len(train_data_augmented))]) -
                    min([train_data_augmented[i]['reward'] for i in range(len(train_data_augmented))])
                )
            )
        for item, rnd_rew in zip(train_data_augmented, rnd_reward):
            if self.intrinsic_reward_type == 'add':
                item['reward'] = item['reward'] + rnd_rew * self.cfg.intrinsic_reward_weight
            elif self.intrinsic_reward_type == 'new':
                item['intrinsic_reward'] = rnd_rew
            elif self.intrinsic_reward_type == 'assign':
                item['reward'] = rnd_rew

        return train_data_augmented

    def collect_data(self, data: list) -> None:
        self.train_obs.extend(collect_states(data))

    def clear_data(self) -> None:
        self.train_obs.clear()

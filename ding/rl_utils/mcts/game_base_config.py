"""
The following code is adapted from https://github.com/YeWR/EfficientZero
"""
import torch
import numpy as np
from ding.model.template.efficientzero.efficientzero_model import EfficientZeroNet


class DiscreteSupport(object):

    def __init__(self, min: int, max: int, delta=1.):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + 1, delta)
        self.size = len(self.range)
        self.delta = delta


class GameBaseConfig(object):

    def __init__(self, game_config=None):
        super(GameBaseConfig, self).__init__()
        self.game_config = game_config
        for k, v in game_config.items():
            self.__dict__[k] = v

    def visit_softmax_temperature_fn(self, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * self.max_training_steps:
                return 1.0
            elif trained_steps < 0.75 * self.max_training_steps:
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_transform(self, x):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        delta = self.value_support.delta
        assert delta == 1
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x / delta) + 1) - 1) + epsilon * x / delta
        return output

    def inverse_reward_transform(self, reward_logits):
        return self.inverse_scalar_transform(reward_logits, self.reward_support)

    def inverse_value_transform(self, value_logits):
        return self.inverse_scalar_transform(value_logits, self.value_support)

    def inverse_scalar_transform(self, logits, scalar_support):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        delta = self.value_support.delta
        value_probs = torch.softmax(logits, dim=1)
        value_support = torch.ones(value_probs.shape)
        value_support[:, :] = torch.from_numpy(np.array([x for x in scalar_support.range]))
        value_support = value_support.to(device=value_probs.device)
        value = (value_support * value_probs).sum(1, keepdim=True) / delta

        epsilon = 0.001
        sign = torch.ones(value.shape).float().to(value.device)
        sign[value < 0] = -1.0
        output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output * delta

        nan_part = torch.isnan(output)
        output[nan_part] = 0.
        output[torch.abs(output) < epsilon] = 0.
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    def _phi(self, x, min, max, set_size: int):
        delta = self.value_support.delta

        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = x - x_low
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min / delta, x_low - min / delta
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

    def get_hparams(self):
        # get all the hyper-parameters
        hparams = {}
        for k, v in self.__dict__.items():
            if 'path' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm
        )


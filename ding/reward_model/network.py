from typing import Union, Tuple, List, Dict, Optional
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from ding.torch_utils.data_helper import to_tensor
from ding.torch_utils import one_hot
from ding.utils.data import default_collate
from .reword_model_utils import concat_state_action_pairs
from functools import partial


class RepresentationNetwork(nn.Module):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        hidden_size_list: SequenceType,
        activation: Optional[nn.Module] = nn.ReLU(),
        kernel_size: Optional[SequenceType] = [8, 4, 3],
        stride: Optional[SequenceType] = [4, 2, 1],
    ) -> None:
        super(RepresentationNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature = FCEncoder(obs_shape, hidden_size_list, activation=activation)
        elif len(obs_shape) == 3:
            self.feature = ConvEncoder(
                obs_shape, hidden_size_list, activation=activation, kernel_size=kernel_size, stride=stride
            )
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own Representation Network".
                format(obs_shape)
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feature_output = self.feature(obs)
        return feature_output


class RNDNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType) -> None:
        super(RNDNetwork, self).__init__()
        self.target = RepresentationNetwork(obs_shape, hidden_size_list)
        self.predictor = RepresentationNetwork(obs_shape, hidden_size_list)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predict_feature = self.predictor(obs)
            target_feature = self.target(obs)
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-8)
        return reward

    def learn(self, obs: torch.Tensor) -> torch.Tensor:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        loss = F.mse_loss(predict_feature, target_feature.detach())
        return loss


class REDNetwork(RNDNetwork):

    def __init__(
            self,
            obs_shape: int,
            action_shape: int,
            hidden_size_list: SequenceType,
            sigma: Optional[float] = 0.5
    ) -> None:
        # RED network does not support high dimension obs
        super().__init__(obs_shape + action_shape, hidden_size_list)
        self.sigma = sigma

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predict_feature = self.predictor(obs)
            target_feature = self.target(obs)
            reward = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            reward = torch.exp(-self.sigma * reward)
        return reward


class GAILNetwork(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            hidden_size_list: SequenceType,
            kernel_size: Optional[SequenceType] = None,
            stride: Optional[SequenceType] = None,
            activation: Optional[nn.Module] = nn.ReLU(),
            action_shape: Optional[int] = None
    ) -> None:
        super(GAILNetwork, self).__init__()
        # Gail will need one more fc layer after RepresentationNetwork, and it will use another activation function
        self.act = nn.Sigmoid()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature = RepresentationNetwork(obs_shape, hidden_size_list, activation)
            self.fc = nn.Linear(hidden_size_list[0], 1)
            self.image_input = False
        elif len(obs_shape) == 3:
            self.action_size = action_shape
            self.obs_size = obs_shape
            self.feature = RepresentationNetwork(obs_shape, hidden_size_list, activation, kernel_size, stride)
            self.fc = nn.Linear(64 + self.action_size, 1)
            self.image_input = True

    def learn(self, train_data: torch.Tensor, expert_data: torch.Tensor) -> torch.Tensor:
        out_1: torch.Tensor = self.forward(train_data)
        loss_1: torch.Tensor = torch.log(out_1 + 1e-8).mean()
        out_2: torch.Tensor = self.forward(expert_data)
        loss_2: torch.Tensor = torch.log(1 - out_2 + 1e-8).mean()

        loss: torch.Tensor = -(loss_1 + loss_2)

        return loss

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.image_input:
            # input: x = [B, 4 x 84 x 84 + self.action_size], last element is action
            actions = data[:, -self.action_size:]  # [B, self.action_size]
            # get observations
            obs = data[:, :-self.action_size]
            obs = obs.reshape([-1] + self.obs_size)  # [B, 4, 84, 84]
            obs = self.feature(obs)
            data = torch.cat([obs, actions], dim=-1)
        else:
            data = self.feature(data)

        data = self.fc(data)
        reward = self.act(data)
        return reward


class ICMNetwork(nn.Module):
    """
    Intrinsic Curiosity Model (ICM Module)
    Implementation of:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf
    [2] Code implementation reference:
    https://github.com/pathak22/noreward-rl
    https://github.com/jcwleo/curiosity-driven-exploration-pytorch

    1) Embedding observations into a latent space
    2) Predicting the action logit given two consecutive embedded observations
    3) Predicting the next embedded obs, given the embeded former observation and action
    """

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType, action_shape: int) -> None:
        super(ICMNetwork, self).__init__()
        self.action_shape = action_shape
        feature_output = hidden_size_list[-1]
        self.feature = RepresentationNetwork(obs_shape, hidden_size_list)
        self.inverse_net = nn.Sequential(nn.Linear(feature_output * 2, 512), nn.ReLU(), nn.Linear(512, action_shape))
        self.residual = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(action_shape + 512, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                ) for _ in range(8)
            ]
        )
        self.forward_net_1 = RepresentationNetwork(action_shape + feature_output, [512], nn.LeakyReLU())
        self.forward_net_2 = nn.Linear(action_shape + 512, feature_output)

    def _forward(self, state: torch.Tensor, next_state: torch.Tensor,
                 action_long: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overview:
            Use observation, next_observation and action to genearte ICM module
            Parameter updates with ICMNetwork forward setup.
        Arguments:
            - state (:obj:`torch.Tensor`):
                The current state batch
            - next_state (:obj:`torch.Tensor`):
                The next state batch
            - action_long (:obj:`torch.Tensor`):
                The action batch
        Returns:
            - real_next_state_feature (:obj:`torch.Tensor`):
                Run with the encoder. Return the real next_state's embedded feature.
            - pred_next_state_feature (:obj:`torch.Tensor`):
                Run with the encoder and residual network. Return the predicted next_state's embedded feature.
            - pred_action_logit (:obj:`torch.Tensor`):
                Run with the encoder. Return the predicted action logit.
        Shapes:
            - state (:obj:`torch.Tensor`): :math:`(B, N)`, where B is the batch size and N is ''obs_shape''
            - next_state (:obj:`torch.Tensor`): :math:`(B, N)`, where B is the batch size and N is ''obs_shape''
            - action_long (:obj:`torch.Tensor`): :math:`(B)`, where B is the batch size''
            - real_next_state_feature (:obj:`torch.Tensor`): :math:`(B, M)`, where B is the batch size
              and M is embedded feature size
            - pred_next_state_feature (:obj:`torch.Tensor`): :math:`(B, M)`, where B is the batch size
              and M is embedded feature size
            - pred_action_logit (:obj:`torch.Tensor`): :math:`(B, A)`, where B is the batch size
              and A is the ''action_shape''
        """
        action = one_hot(action_long, num=self.action_shape)
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action logit
        concat_state = torch.cat((encode_state, encode_next_state), 1)
        pred_action_logit = self.inverse_net(concat_state)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)
            ) + pred_next_state_feature_orig
        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))
        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action_logit

    def learn(self, state: torch.Tensor, next_state: torch.Tensor,
              action_long: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        real_next_state_feature, pred_next_state_feature, pred_action_logit = self._forward(
            state, next_state, action_long
        )

        inverse_loss = F.cross_entropy(pred_action_logit, action_long.long())
        forward_loss = F.mse_loss(pred_next_state_feature, real_next_state_feature.detach()).mean()
        action = torch.argmax(F.softmax(pred_action_logit), -1)
        accuracy = torch.sum(action == action_long.squeeze(-1)).item() / action_long.shape[0]
        return inverse_loss, forward_loss, accuracy

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action_long: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            real_next_state_feature, pred_next_state_feature, _ = self._forward(state, next_state, action_long)
            reward = F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction="none").mean(dim=1)

        return reward


class GCLNetwork(nn.Module):

    def __init__(
            self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType, output_size: int,
            action_shape: int
    ) -> None:
        super(GCLNetwork, self).__init__()
        self.feature = RepresentationNetwork(obs_shape, hidden_size_list)
        self.fc = nn.Linear(hidden_size_list[-1], output_size)
        self.action_shape = action_shape

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        reward = self.feature(data)
        reward = self.fc(reward)

        return reward

    def learn(self, expert_demo: torch.Tensor, samp: torch.Tensor) -> torch.Tensor:
        cost_demo = self.forward(
            torch.cat([expert_demo['obs'], expert_demo['action'].float().reshape(-1, self.action_shape)], dim=-1)
        )
        cost_samp = self.forward(
            torch.cat([samp['obs'], samp['action'].float().reshape(-1, self.action_shape)], dim=-1)
        )
        prob = samp['prob'].unsqueeze(-1)
        loss_IOC = torch.mean(cost_demo) + \
            torch.log(torch.mean(torch.exp(-cost_samp)/(prob+1e-7)))

        return loss_IOC


class TREXNetwork(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            hidden_size_list: SequenceType,
            kernel_size: Optional[SequenceType] = None,
            stride: Optional[SequenceType] = None,
            activation: Optional[nn.Module] = nn.ReLU(),
            l1_reg: Optional[float] = 0,
    ) -> None:
        super(TREXNetwork, self).__init__()
        self.input_size = obs_shape
        self.l1_reg = l1_reg
        self.output_size = hidden_size_list[-1]
        hidden_size_list = hidden_size_list[:-1]
        self.feature = RepresentationNetwork(obs_shape, hidden_size_list, activation, kernel_size, stride)
        self.act = activation
        self.fc = nn.Linear(hidden_size_list[-1], self.output_size)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        reward = self.feature(data)
        if isinstance(self.input_size, int) is False and len(self.input_size) == 3:
            reward = self.act(reward)
        reward = self.fc(reward)
        return reward

    def learn(self, traj_i: torch.Tensor, traj_j: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs, total_abs_reward = self.get_outputs_abs_reward(traj_i, traj_j)
        outputs = outputs.unsqueeze(0)
        loss = F.cross_entropy(outputs, labels) + self.l1_reg * total_abs_reward
        return loss

    def get_outputs_abs_reward(self, traj_i: torch.Tensor, traj_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reward_i = self.forward(traj_i)
        reward_j = self.forward(traj_j)

        cum_r_i = torch.sum(reward_i)
        cum_r_j = torch.sum(reward_j)
        outputs = torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0)
        total_abs_reward = torch.sum(torch.abs(reward_i)) + torch.sum(torch.abs(reward_j))

        return outputs, total_abs_reward

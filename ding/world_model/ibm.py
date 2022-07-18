from tokenize import String
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from typing import Union, Optional, Dict

from ding.model.common.head import DiscreteHead, RegressionHead, ReparameterizationHead
from ding.utils import SequenceType, squeeze
from ding.model.common.encoder import FCEncoder, ConvEncoder
from torch.distributions import Independent, Normal


class InverseDynamicsModel(nn.Module):
    """
    InverseDynamicsModel: infering missing action information from state transition.
    input and output: given pair of observation, return action (s0,s1 --> a0 if n=2)
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            continuous: Boolean,
            encoder_hidden_size_list: SequenceType = [60, 80, 100, 40],
            action_space: String = "regression",
            activation: Optional[nn.Module] = nn.LeakyReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the Inverse Dynamics (encoder + head) Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - continuous(:obj:`Boolean`): whether action is continuous. eg: continuous:True, discrete: False.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.LeakyReLU()`` refer to https://arxiv.org/abs/1805.01954
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details.
        """
        super(InverseDynamicsModel, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        # FC encoder: obs and obs[next] ,so input shape is obs_shape*2
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(
                obs_shape * 2, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
        elif len(obs_shape) == 3:
            obs_shape[0] = obs_shape[0] * 2
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own Model".format(obs_shape)
            )
        self.continuous = continuous
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization']
        if self.continuous:
            if self.action_space == "regression":
                self.header = RegressionHead(
                    encoder_hidden_size_list[-1],
                    action_shape,
                    final_tanh=False,
                    activation=activation,
                    norm_type=norm_type
                )
            elif self.action_space == "reparameterization":
                self.header = ReparameterizationHead(
                    encoder_hidden_size_list[-1],
                    action_shape,
                    sigma_type='conditioned',
                    activation=activation,
                    norm_type=norm_type
                )
        else:
            self.header = DiscreteHead(
                encoder_hidden_size_list[-1], action_shape, activation=activation, norm_type=norm_type
            )

    def forward(self, x: torch.Tensor) -> Dict:
        if self.continuous:
            if self.action_space == "regression":
                x = self.encoder(x)
                x = self.header(x)
                return {'action': x['pred']}
            elif self.action_space == "reparameterization":
                x = self.encoder(x)
                x = self.header(x)
                mu, sigma = x['mu'], x['sigma']
                dist = Independent(Normal(mu, sigma), 1)
                pred = dist.rsample()
                action = torch.tanh(pred)
                return {'logit': [mu, sigma], 'action': action}
        else:
            x = self.encoder(x)
            x = self.header(x)
            return x

    def predict_action(self, x: torch.Tensor) -> Dict:
        if self.continuous:
            return self.forward(x)
        else:
            res = nn.Softmax(dim=-1)
            action = torch.argmax(res(self.forward(x)['logit']), -1)
            return {'action': action}

    def train(self, training_set, n_epoch, learning_rate, weight_decay):
        '''
        train transition model, given pair of states return action (s0,s1 ---> a0)
        Input:
        training_set: states transition
        n_epoch: number of epoches
        learning_rate: learning rate for optimizer
        weight_decay: weight decay for optimizer
        return:
        loss: trained transition model
        '''
        if self.continuous:
            # criterion = nn.MSELoss()
            criterion = nn.L1Loss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_list = []
        for itr in range(n_epoch):
            data = training_set['obs']
            y = training_set['action']
            if self.continuous:
                y_pred = self.forward(data)['action']
            else:
                y_pred = self.forward(data)['logit']
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        last_loss = loss_list[-1]
        return last_loss

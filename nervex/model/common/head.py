from typing import Union, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nervex.torch_utils import fc_block, noise_block, NoiseLinearLayer, MLP
from typing import Callable
from nervex.rl_utils import beta_function_map


class BaseHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        action_shape: int,
        layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - action_shape (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(BaseHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.Q = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, action_shape)
        )

    def forward(self, x: torch.Tensor) -> Dict:
        logit = self.Q(x)
        return {'logit': logit}


class DistributionHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        action_shape: int,
        layer_num: int = 1,
        n_atom: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - action_shape (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(DistributionHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.Q = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, action_shape * n_atom)
        )
        self.action_shape = action_shape
        self.n_atom = n_atom
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, x: torch.Tensor) -> Dict:
        q = self.Q(x)
        q = q.view(*q.shape[:-1], self.action_shape, self.n_atom)
        dist = torch.softmax(q, dim=-1) + 1e-6
        q = dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(x)
        q = q.sum(-1)
        return {'logit': q, 'distribution': dist}


class RainbowHead(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            action_shape: int,
            layer_num: int = 1,
            n_atom: int = 51,
            v_min: float = -10,
            v_max: float = 10,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            noise: bool = True,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - action_shape (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(RainbowHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.A = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, action_shape * n_atom)
        )
        self.Q = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, n_atom)
        )
        self.action_shape = action_shape
        self.n_atom = n_atom
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, x: torch.Tensor) -> Dict:
        a = self.A(x)
        q = self.Q(x)
        a = a.view(*a.shape[:-1], self.action_shape, self.n_atom)
        q = q.view(*q.shape[:-1], 1, self.n_atom)
        q = q + a - a.mean(dim=-2, keepdim=True)
        dist = torch.softmax(q, dim=-1) + 1e-6
        q = dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(x)
        q = q.sum(-1)
        return {'logit': q, 'distribution': dist}


class QRDQNHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        action_shape: int,
        layer_num: int = 1,
        num_quantiles: int = 32,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - action_shape (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(QRDQNHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.Q = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, action_shape * num_quantiles)
        )
        self.num_quantiles = num_quantiles
        self.action_shape = action_shape

    def forward(self, x: torch.Tensor) -> Dict:
        q = self.Q(x)
        q = q.view(*q.shape[:-1], self.action_shape, self.num_quantiles)

        logit = q.mean(-1)
        tau = torch.linspace(0, 1, self.num_quantiles + 1)
        tau = ((tau[:-1] + tau[1:]) / 2).view(1, -1, 1).repeat(q.shape[0], 1, 1).to(q)
        return {'logit': logit, 'q': q, 'tau': tau}


class QuantileHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        action_shape: int,
        layer_num: int = 1,
        num_quantiles: int = 32,
        quantile_embedding_size: int = 128,
        beta_function_type: str = 'uniform',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - action_shape (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(QuantileHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.Q = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, action_shape)
        )
        self.num_quantiles = num_quantiles
        self.quantile_embedding_size = quantile_embedding_size
        self.action_shape = action_shape
        self.iqn_fc = nn.Linear(self.quantile_embedding_size, hidden_size)
        self.beta_function = beta_function_map[beta_function_type]

    def quantile_net(self, quantiles: torch.Tensor) -> torch.Tensor:
        quantile_net = quantiles.repeat([1, self.quantile_embedding_size])
        quantile_net = torch.cos(
            torch.arange(1, self.quantile_embedding_size + 1, 1).to(quantiles) * math.pi * quantile_net
        )
        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = F.relu(quantile_net)
        return quantile_net

    def forward(self, x: torch.Tensor, num_quantiles: int = None) -> Dict:
        if num_quantiles is None:
            num_quantiles = self.num_quantiles
        batch_size = x.shape[0]

        q_quantiles = torch.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1).to(x)
        logit_quantiles = torch.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1).to(x)
        logit_quantiles = self.beta_function(logit_quantiles)
        q_quantile_net = self.quantile_net(q_quantiles)
        logit_quantile_net = self.quantile_net(logit_quantiles)

        x = x.repeat(num_quantiles, 1)
        q_x = x * q_quantile_net
        logit_x = x * logit_quantile_net

        q = self.Q(q_x).reshape(num_quantiles, batch_size, -1)
        logit = self.Q(logit_x).reshape(num_quantiles, batch_size, -1).mean(0)

        return {'logit': logit, 'q': q, 'quantiles': q_quantiles}


class DuelingHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        action_shape: int,
        a_layer_num: int = 1,
        v_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - action_shape (:obj:`int`): the num of actions
            - a_layer_num (:obj:`int`): the num of fc_block used in the network to compute action output
            - v_layer_num (:obj:`int`): the num of fc_block used in the network to compute value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(DuelingHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.A = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                a_layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, action_shape)
        )
        self.V = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                v_layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict:
        a = self.A(x)
        v = self.V(x)
        logit = a - a.mean(dim=-1, keepdim=True) + v
        return {'logit': logit}


head_fn_map = {
    'base': BaseHead,
    'dueling': DuelingHead,
    'distribution': DistributionHead,
    'rainbow': RainbowHead,
    'qrdqn': QRDQNHead,
    'quantile': QuantileHead,
}

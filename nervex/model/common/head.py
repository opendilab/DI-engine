from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nervex.torch_utils import fc_block, noise_block, NoiseLinearLayer, mlp
from typing import Callable
from nervex.rl_utils import beta_function_map

class BaseHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        init_type: Optional[str] = "xavier",
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_dim (:obj:`int`): the hidden_dim used before connected to DuelingHead
            - action_dim (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(BaseHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.Q = nn.Sequential(mlp(hidden_dim, hidden_dim, hidden_dim, layer_num, layer_fn=layer, init_type=init_type, activation=activation, norm_type=norm_type),
                                block(hidden_dim, action_dim, init_type=init_type))

    def forward(self, x: torch.Tensor) -> Dict:
        logit = self.Q(x)
        return {'logit': logit}

class DistributionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        layer_num: int = 1,
        n_atom: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        init_type: Optional[str] = "xavier",
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_dim (:obj:`int`): the hidden_dim used before connected to DuelingHead
            - action_dim (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(DistributionHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.Q = nn.Sequential(mlp(hidden_dim, hidden_dim, hidden_dim, layer_num, layer_fn=layer, init_type=init_type, activation=activation, norm_type=norm_type),
                                block(hidden_dim, action_dim * n_atom, init_type=init_type))
        self.action_dim = action_dim
        self.n_atom = n_atom
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, x: torch.Tensor) -> Dict:
        q = self.Q(x)
        q = q.view(*q.shape[:-1], self.action_dim, self.n_atom)
        dist = torch.softmax(q, dim=-1) + 1e-6
        q = dist * torch.linspace(self.v_min, self.v_max,
                                  self.n_atom).to(torch.device("cuda" if dist.is_cuda else "cpu"))
        q = q.sum(-1)
        return {'logit': q, 'distribution': dist}

class QuantileHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        layer_num: int = 1,
        num_quantiles: int = 32,
        quantile_embedding_dim: int = 128,
        beta_function_type: str = 'uniform',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        init_type: Optional[str] = "xavier",
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_dim (:obj:`int`): the hidden_dim used before connected to DuelingHead
            - action_dim (:obj:`int`): the num of actions
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(QuantileHead, self).__init__()
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.Q = nn.Sequential(mlp(hidden_dim, hidden_dim, hidden_dim, layer_num, layer_fn=layer, init_type=init_type, activation=activation, norm_type=norm_type),
                                block(hidden_dim, action_dim, init_type=init_type))
        self.num_quantiles = num_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim
        self.action_dim = action_dim
        self.iqn_fc = nn.Linear(self.quantile_embedding_dim, hidden_dim)
        self.beta_function = beta_function_map[beta_function_type]

    def quantile_net(self, quantiles: torch.Tensor) -> torch.Tensor:
        quantile_net = quantiles.repeat([1, self.quantile_embedding_dim])
        quantile_net = torch.cos(
            torch.arange(1, self.quantile_embedding_dim + 1, 1).to(quantiles) * math.pi *
            quantile_net
        )
        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = F.relu(quantile_net)
        return quantile_net

    def forward(self, x: torch.Tensor) -> Dict:
        batch_size = x.shape[0]
        device = torch.device("cuda" if x.is_cuda else "cpu")

        q_quantiles = torch.FloatTensor(self.num_quantiles * batch_size, 1).uniform_(0, 1).to(device)
        logit_quantiles = torch.FloatTensor(self.num_quantiles * batch_size, 1).uniform_(0, 1).to(device)
        logit_quantiles = self.beta_function(logit_quantiles)
        q_quantile_net = self.quantile_net(q_quantiles)
        logit_quantile_net = self.quantile_net(logit_quantiles)

        x = x.repeat(num_quantiles, 1)
        q_x = x * q_quantile_net
        logit_x = x * logit_quantile_net

        q = self.Q(q_x).reshape(self.num_quantiles, batch_size, -1)
        logit = self.Q(logit_x).reshape(self.num_quantiles, batch_size, -1).mean(0)

        return {'logit': logit, 'q': q, 'quantiles': quantiles}

class DuelingHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        a_layer_num: int = 1,
        v_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        init_type: Optional[str] = "xavier",
        noise: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_dim (:obj:`int`): the hidden_dim used before connected to DuelingHead
            - action_dim (:obj:`int`): the num of actions
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
        self.A = nn.Sequential(mlp(hidden_dim, hidden_dim, hidden_dim, a_layer_num, layer_fn=layer, init_type=init_type, activation=activation, norm_type=norm_type),
                                block(hidden_dim, action_dim, init_type=init_type))
        self.V = nn.Sequential(mlp(hidden_dim, hidden_dim, hidden_dim, v_layer_num, layer_fn=layer, init_type=init_type, activation=activation, norm_type=norm_type),
                                block(hidden_dim, 1, init_type=init_type))

    def forward(self, x: torch.Tensor) -> Dict:
        a = self.A(x)
        v = self.V(x)
        logit = a - a.mean(dim=-1, keepdim=True) + v
        return {'logit': logit}

head_fn_map = {
    'base': BaseHead,
    'dueling': DuelingHead,
    'distribution': DistributionHead,
    'quantile': QuantileHead,
}

from typing import Union, Optional, Tuple, Dict, Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nervex.torch_utils import fc_block, noise_block, NoiseLinearLayer, MLP
from nervex.rl_utils import beta_function_map
from nervex.utils import lists_to_dicts, SequenceType


class ClassificationHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - output_size (:obj:`int`): the number of output
            - layer_num (:obj:`int`): the num of fc_block used in the network to compute Q value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(ClassificationHead, self).__init__()
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
            ), block(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``
        """
        logit = self.Q(x)
        return {'logit': logit}


class DistributionHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        n_atom: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
        eps: Optional[float] = 1e-6,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - output_size (:obj:`int`): the num of output
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
            ), block(hidden_size, output_size * n_atom)
        )
        self.output_size = output_size
        self.n_atom = n_atom
        self.v_min = v_min
        self.v_max = v_max
        self.eps = eps  # for numerical stability

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``, ``distribution``
        """
        q = self.Q(x)
        q = q.view(*q.shape[:-1], self.output_size, self.n_atom)
        dist = torch.softmax(q, dim=-1) + self.eps
        q = dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(x)
        q = q.sum(-1)
        return {'logit': q, 'distribution': dist}


class RainbowHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        n_atom: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = True,
        eps: Optional[float] = 1e-6,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - output_size (:obj:`int`): the num of output
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
            ), block(hidden_size, output_size * n_atom)
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
        self.output_size = output_size
        self.n_atom = n_atom
        self.v_min = v_min
        self.v_max = v_max
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``, ``distribution``
        """
        a = self.A(x)
        q = self.Q(x)
        a = a.view(*a.shape[:-1], self.output_size, self.n_atom)
        q = q.view(*q.shape[:-1], 1, self.n_atom)
        q = q + a - a.mean(dim=-2, keepdim=True)
        dist = torch.softmax(q, dim=-1) + self.eps
        q = dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(x)
        q = q.sum(-1)
        return {'logit': q, 'distribution': dist}


class QRDQNHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        num_quantiles: int = 32,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - output_size (:obj:`int`): the num of output
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
            ), block(hidden_size, output_size * num_quantiles)
        )
        self.num_quantiles = num_quantiles
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``,  ``q``, ``tau``
        """
        q = self.Q(x)
        q = q.view(*q.shape[:-1], self.output_size, self.num_quantiles)

        logit = q.mean(-1)
        tau = torch.linspace(0, 1, self.num_quantiles + 1)
        tau = ((tau[:-1] + tau[1:]) / 2).view(1, -1, 1).repeat(q.shape[0], 1, 1).to(q)
        return {'logit': logit, 'q': q, 'tau': tau}


class QuantileHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        num_quantiles: int = 32,
        quantile_embedding_size: int = 128,
        beta_function_type: Optional[str] = 'uniform',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - output_size (:obj:`int`): the num of output
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
            ), block(hidden_size, output_size)
        )
        self.num_quantiles = num_quantiles
        self.quantile_embedding_size = quantile_embedding_size
        self.output_size = output_size
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

    def forward(self, x: torch.Tensor, num_quantiles: Optional[int] = None) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``, ``q``, ``quantiles``
        """
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
        output_size: int,
        layer_num: int = 1,
        a_layer_num: Optional[int] = None,
        v_layer_num: Optional[int] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): the hidden_size used before connected to DuelingHead
            - output_size (:obj:`int`): the num of output
            - a_layer_num (:obj:`int`): the num of fc_block used in the network to compute action output
            - v_layer_num (:obj:`int`): the num of fc_block used in the network to compute value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - noise (:obj:`bool`): whether use noisy fc block
        """
        super(DuelingHead, self).__init__()
        if a_layer_num is None:
            a_layer_num = layer_num
        if v_layer_num is None:
            v_layer_num = layer_num
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
            ), block(hidden_size, output_size)
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
        """
        ReturnsKeys:
            - necessary: ``logit``
        """
        a = self.A(x)
        v = self.V(x)
        logit = a - a.mean(dim=-1, keepdim=True) + v
        return {'logit': logit}


class RegressionHead(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            layer_num: int = 2,
            final_tanh: Optional[bool] = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(RegressionHead, self).__init__()
        self.main = MLP(hidden_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.last = nn.Linear(hidden_size, output_size)  # for convenience of special initialization
        self.final_tanh = final_tanh
        if self.final_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``pred``
        """
        x = self.main(x)
        x = self.last(x)
        if self.final_tanh:
            x = self.tanh(x)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return {'pred': x}


class ReparameterizationHead(nn.Module):
    default_sigma_type = ['fixed', 'independent', 'conditioned']

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            layer_num: int = 2,
            sigma_type: Optional[str] = None,
            fixed_sigma_value: Optional[float] = 1.0,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(ReparameterizationHead, self).__init__()
        self.sigma_type = sigma_type
        assert sigma_type in self.default_sigma_type, "Please indicate sigma_type as one of {}".format(
            self.default_sigma_type
        )
        self.main = MLP(hidden_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.mu = nn.Linear(hidden_size, output_size)
        if self.sigma_type == 'fixed':
            self.sigma = torch.full((1, output_size), fixed_sigma_value)
        elif self.sigma_type == 'independent':  # independent parameter
            self.log_sigma_param = nn.Parameter(torch.zeros(1, output_size))
        elif self.sigma_type == 'conditioned':
            self.log_sigma_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``mu``, ``sigma``
        """
        x = self.main(x)
        mu = self.mu(x)
        if self.sigma_type == 'fixed':
            sigma = self.sigma.to(mu.device) + torch.zeros_like(mu)  # addition aims to broadcast shape
        elif self.sigma_type == 'independent':
            log_sigma = self.log_sigma_param + torch.zeros_like(mu)  # addition aims to broadcast shape
            sigma = torch.exp(log_sigma)
        elif self.sigma_type == 'conditioned':
            log_sigma = self.log_sigma_layer(x)
            sigma = torch.exp(torch.clamp(log_sigma, -20, 2))
        return {'mu': mu, 'sigma': sigma}


class MultiDiscreteHead(nn.Module):

    def __init__(self, head_cls: type, hidden_size: int, output_size_list: SequenceType, **head_kwargs) -> None:
        r"""
        Overview:
            Init the MultiDiscreteHead according to arguments.
        Arguments:
            - head_cls (:obj:`type`): The class of head, like dueling_head, distribution_head, quatile_head, etc
            - hidden_size (:obj:`int`): The number of hidden layer size
            - output_size_list (:obj:`int`): The collection of output_size, e.g.: multi discrete action, [2, 3, 5]
            - head_kwargs: (:boj:`dict`): Class-specific arguments
        """
        super(MultiDiscreteHead, self).__init__()
        self.pred = nn.ModuleList()
        for size in output_size_list:
            self.pred.append(head_cls(hidden_size, size, **head_kwargs))

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict multi discrete output
        Arguments:
            - x (:obj:`torch.Tensor`): The encoded embedding tensor, usually with shape (B, N)
        Returns:
            - return (:obj:`Dict`): Prediction output dict
        Examples:
            >>> head = MultiDiscreteHead(DuelingHead, 64, [2, 3, 5], v_layer_num=2)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'][0].shape == (4, 2)
        """
        return lists_to_dicts([m(x) for m in self.pred])


head_cls_map = {
    # discrete
    'classification': ClassificationHead,
    'dueling': DuelingHead,
    'distribution': DistributionHead,
    'rainbow': RainbowHead,
    'qrdqn': QRDQNHead,
    'quantile': QuantileHead,
    # continuous
    'regression': RegressionHead,
    'reparameterization': ReparameterizationHead,
}

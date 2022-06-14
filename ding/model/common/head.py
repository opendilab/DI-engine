from typing import Optional, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import fc_block, noise_block, NoiseLinearLayer, MLP
from ding.rl_utils import beta_function_map
from ding.utils import lists_to_dicts, SequenceType


class DiscreteHead(nn.Module):
    """
        Overview:
            The ``DiscreteHead`` used to output discrete actions logit. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``logit``.
        Interfaces:
            ``__init__``, ``forward``.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
    ) -> None:
        """
        Overview:
            Init the ``DiscreteHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``DiscreteHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value output.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
        """
        super(DiscreteHead, self).__init__()
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
        Overview:
            Use encoded embedding tensor to run MLP with ``DiscreteHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keyword ``logit`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.

        Examples:
            >>> head = DiscreteHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 64])
        """
        logit = self.Q(x)
        return {'logit': logit}


class DistributionHead(nn.Module):
    """
        Overview:
            The ``DistributionHead`` used to output Q-value distribution. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
        outputs ``logit`` and ``distribution``.

        Interfaces:
            ``__init__``, ``forward``.
    """

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
        """
        Overview:
            Init the ``DistributionHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``DistributionHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value distribution.
            - n_atom (:obj:`int`): The number of atoms (discrete supports). Default is ``51``.
            - v_min (:obj:`int`): Min value of atoms. Default is ``-10``.
            - v_max (:obj:`int`): Max value of atoms. Default is ``10``.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
            - eps (:obj:`float`): Small constant used for numerical stability.
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
        Overview:
            Use encoded embedding tensor to run MLP with ``DistributionHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``logit`` (:obj:`torch.Tensor`) and \
                ``distribution`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.
            - distribution: :math:`(B, M, n_atom)`.

        Examples:
            >>> head = DistributionHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default n_atom is 51
            >>> assert outputs['distribution'].shape == torch.Size([4, 64, 51])
        """
        q = self.Q(x)
        q = q.view(*q.shape[:-1], self.output_size, self.n_atom)
        dist = torch.softmax(q, dim=-1) + self.eps
        q = dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(x)
        q = q.sum(-1)
        return {'logit': q, 'distribution': dist}


class RainbowHead(nn.Module):
    """
        Overview:
            The ``RainbowHead`` used to output Q-value distribution. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            outputs ``logit`` and ``distribution``.
        Interfaces:
            ``__init__``, ``forward``.
    """

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
        """
        Overview:
            Init the ``RainbowHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``RainbowHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value output.
            - n_atom (:obj:`int`): The number of atoms (discrete supports). Default is ``51``.
            - v_min (:obj:`int`): Min value of atoms. Default is ``-10``.
            - v_max (:obj:`int`): Max value of atoms. Default is ``10``.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
            - eps (:obj:`float`): Small constant used for numerical stability.
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
        Overview:
            Use encoded embedding tensor to run MLP with ``RainbowHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``logit`` (:obj:`torch.Tensor`) and \
                ``distribution`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.
            - distribution: :math:`(B, M, n_atom)`.

        Examples:
            >>> head = RainbowHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default n_atom is 51
            >>> assert outputs['distribution'].shape == torch.Size([4, 64, 51])
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
    """
        Overview:
            The ``QRDQNHead`` (Quantile Regression DQN) used to output action quantiles. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``logit``, ``q``, and ``tau``.
        Interfaces:
            ``__init__``, ``forward``.
    """

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
        """
        Overview:
            Init the ``QRDQNHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``QRDQNHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value output.
            - num_quantiles (:obj:`int`): The number of quantiles. Default is ``32``.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
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
        Overview:
            Use encoded embedding tensor to run MLP with ``QRDQNHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``logit`` (:obj:`torch.Tensor`), \
                ``q`` (:obj:`torch.Tensor`), and ``tau`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.
            - q: :math:`(B, M, num_quantiles)`.
            - tau: :math:`(B, M, 1)`.

        Examples:
            >>> head = QRDQNHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles is 32
            >>> assert outputs['q'].shape == torch.Size([4, 64, 32])
            >>> assert outputs['tau'].shape == torch.Size([4, 32, 1])
        """
        q = self.Q(x)
        q = q.view(*q.shape[:-1], self.output_size, self.num_quantiles)

        logit = q.mean(-1)
        tau = torch.linspace(0, 1, self.num_quantiles + 1)
        tau = ((tau[:-1] + tau[1:]) / 2).view(1, -1, 1).repeat(q.shape[0], 1, 1).to(q)
        return {'logit': logit, 'q': q, 'tau': tau}


class QuantileHead(nn.Module):
    """
        Overview:
            The ``QuantileHead`` used to output action quantiles. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``logit``, ``q``, and ``quantiles``.
        Interfaces:
            ``__init__``, ``forward``, ``quantile_net``.
    """

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
        """
        Overview:
            Init the ``QuantileHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``QuantileHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value output.
            - num_quantiles (:obj:`int`): The number of quantiles.
            - quantile_embedding_size (:obj:`int`): The embedding size of a quantile.
            - beta_function_type (:obj:`str`): Type of beta function. See ``ding.rl_utils.beta_function.py`` \
                for more details. Default is ``uniform``.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
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
        """
        Overview:
           Deterministic parametric function trained to reparameterize samples from a base distribution. \
           By repeated Bellman update iterations of Q-learning, the optimal action-value function is estimated.
        Arguments:
            - x (:obj:`torch.Tensor`): The encoded embedding tensor of parametric sample.
        Returns:
            - quantile_net (:obj:`torch.Tensor`): Quantile network output tensor after reparameterization.
        Shapes:
            - quantile_net :math:`(quantile_embedding_size, M)`, where ``M = output_size``.
        Examples:
            >>> head = QuantileHead(64, 64)
            >>> quantiles = torch.randn(128,1)
            >>> qn_output = head.quantile_net(quantiles)
            >>> assert isinstance(qn_output, torch.Tensor)
            >>> # default quantile_embedding_size: int = 128,
            >>> assert qn_output.shape == torch.Size([128, 64])
        """
        quantile_net = quantiles.repeat([1, self.quantile_embedding_size])
        quantile_net = torch.cos(
            torch.arange(1, self.quantile_embedding_size + 1, 1).to(quantiles) * math.pi * quantile_net
        )
        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = F.relu(quantile_net)
        return quantile_net

    def forward(self, x: torch.Tensor, num_quantiles: Optional[int] = None) -> Dict:
        """
        Overview:
            Use encoded embedding tensor to run MLP with ``QuantileHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``logit`` (:obj:`torch.Tensor`), \
                ``q`` (:obj:`torch.Tensor`), and ``quantiles`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.
            - q: :math:`(num_quantiles, B, M)`.
            - quantiles: :math:`(quantile_embedding_size, 1)`.

        Examples:
            >>> head = QuantileHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles is 32
            >>> assert outputs['q'].shape == torch.Size([32, 4, 64])
            >>> assert outputs['quantiles'].shape == torch.Size([128, 1])
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
        q_x = x * q_quantile_net  # 4*32,64
        logit_x = x * logit_quantile_net

        q = self.Q(q_x).reshape(num_quantiles, batch_size, -1)
        logit = self.Q(logit_x).reshape(num_quantiles, batch_size, -1).mean(0)

        return {'logit': logit, 'q': q, 'quantiles': q_quantiles}


class FQFHead(nn.Module):
    """
        Overview:
            The ``FQFHead`` used to output action quantiles. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``logit``, ``q``, ``quantiles``, ``quantiles_hats``, ``q_tau_i`` and ``entropies``.
        Interfaces:
            ``__init__``, ``forward``, ``quantile_net``.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        num_quantiles: int = 32,
        quantile_embedding_size: int = 128,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
    ) -> None:
        """
        Overview:
            Init the ``FQFHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``FQFHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value output.
            - num_quantiles (:obj:`int`): The number of quantiles.
            - quantile_embedding_size (:obj:`int`): The embedding size of a quantile.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
        """
        super(FQFHead, self).__init__()
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
        self.fqf_fc = nn.Sequential(nn.Linear(self.quantile_embedding_size, hidden_size), nn.ReLU())
        self.register_buffer(
            'sigma_pi',
            torch.arange(1, self.quantile_embedding_size + 1, 1).view(1, 1, self.quantile_embedding_size) * math.pi
        )
        # initialize weights_xavier of quantiles_proposal network
        quantiles_proposal_fc = nn.Linear(hidden_size, num_quantiles)
        torch.nn.init.xavier_uniform_(quantiles_proposal_fc.weight, gain=0.01)
        torch.nn.init.constant_(quantiles_proposal_fc.bias, 0)
        self.quantiles_proposal = nn.Sequential(quantiles_proposal_fc, nn.LogSoftmax(dim=1))

    def quantile_net(self, quantiles: torch.Tensor) -> torch.Tensor:
        """
        Overview:
           Deterministic parametric function trained to reparameterize samples from the quantiles_proposal network. \
           By repeated Bellman update iterations of Q-learning, the optimal action-value function is estimated.
        Arguments:
            - x (:obj:`torch.Tensor`): The encoded embedding tensor of parametric sample.
        Returns:
            - quantile_net (:obj:`torch.Tensor`): Quantile network output tensor after reparameterization.
        Examples:
            >>> head = FQFHead(64, 64)
            >>> quantiles = torch.randn(4,32)
            >>> qn_output = head.quantile_net(quantiles)
            >>> assert isinstance(qn_output, torch.Tensor)
            >>> # default quantile_embedding_size: int = 128,
            >>> assert qn_output.shape == torch.Size([4, 32, 64])
        """
        batch_size, num_quantiles = quantiles.shape[:2]
        quantile_net = torch.cos(self.sigma_pi.to(quantiles) * quantiles.view(batch_size, num_quantiles, 1))
        quantile_net = self.fqf_fc(quantile_net)  # (batch_size, num_quantiles, hidden_size)
        return quantile_net

    def forward(self, x: torch.Tensor, num_quantiles: Optional[int] = None) -> Dict:
        """
        Overview:
            Use encoded embedding tensor to run MLP with ``FQFHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``logit`` (:obj:`torch.Tensor`), \
                ``q`` (:obj:`torch.Tensor`), ``quantiles`` (:obj:`torch.Tensor`), \
                ``quantiles_hats`` (:obj:`torch.Tensor`), \
                ``q_tau_i`` (:obj:`torch.Tensor`), ``entropies`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.
            - q: :math:`(B, num_quantiles, M)`.
            - quantiles: :math:`(B, num_quantiles + 1)`.
            - quantiles_hats: :math:`(B, num_quantiles)`.
            - q_tau_i: :math:`(B, num_quantiles - 1, M)`.
            - entropies: :math:`(B, 1)`.
        Examples:
            >>> head = FQFHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles is 32
            >>> assert outputs['q'].shape == torch.Size([4, 32, 64])
            >>> assert outputs['quantiles'].shape == torch.Size([4, 33])
            >>> assert outputs['quantiles_hats'].shape == torch.Size([4, 32])
            >>> assert outputs['q_tau_i'].shape == torch.Size([4, 31, 64])
            >>> assert outputs['quantiles'].shape == torch.Size([4, 1])
        """

        if num_quantiles is None:
            num_quantiles = self.num_quantiles
        batch_size = x.shape[0]

        log_q_quantiles = self.quantiles_proposal(
            x.detach()
        )  # (batch_size, num_quantiles), not to update encoder when learning w1_loss(fraction loss)
        q_quantiles = log_q_quantiles.exp()

        # Calculate entropies of value distributions.
        entropies = -(log_q_quantiles * q_quantiles).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        assert entropies.shape == (batch_size, 1)

        # accumalative softmax
        q_quantiles = torch.cumsum(q_quantiles, dim=1)

        # quantile_hats: find the optimal condition for τ to minimize W1(Z, τ)
        tau_0 = torch.zeros((batch_size, 1)).to(x)
        q_quantiles = torch.cat((tau_0, q_quantiles), dim=1)  # [batch_size, num_quantiles+1]

        q_quantiles_hats = (q_quantiles[:, 1:] + q_quantiles[:, :-1]).detach() / 2.  # (batch_size, num_quantiles)

        q_quantile_net = self.quantile_net(q_quantiles_hats)  # [batch_size, num_quantiles, hidden_size(64)]
        # x.view[batch_size, 1, hidden_size(64)]
        q_x = (x.view(batch_size, 1, -1) * q_quantile_net)  # [batch_size, num_quantiles, hidden_size(64)]

        q = self.Q(q_x)  # [batch_size, num_quantiles, action_dim(64)]

        logit = q.mean(1)
        with torch.no_grad():
            q_tau_i_net = self.quantile_net(
                q_quantiles[:, 1:-1].detach()
            )  # [batch_size, num_quantiles-1, hidden_size(64)]
            q_tau_i_x = (x.view(batch_size, 1, -1) * q_tau_i_net)  # [batch_size, (num_quantiles-1), hidden_size(64)]

            q_tau_i = self.Q(q_tau_i_x)  # [batch_size, num_quantiles-1, action_dim(64)]

        return {
            'logit': logit,
            'q': q,
            'quantiles': q_quantiles,
            'quantiles_hats': q_quantiles_hats,
            'q_tau_i': q_tau_i,
            'entropies': entropies
        }


class DuelingHead(nn.Module):
    """
        Overview:
            The ``DuelingHead`` used to output discrete actions logit. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``logit``.
        Interfaces:
            ``__init__``, ``forward``.
    """

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
        """
        Overview:
            Init the ``DuelingHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``DuelingHead``.
            - output_size (:obj:`int`): The number of outputs.
            - a_layer_num (:obj:`int`): The number of layers used in the network to compute action output.
            - v_layer_num (:obj:`int`): The number of layers used in the network to compute value output.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
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
        Overview:
            Use encoded embedding tensor to run MLP with ``DuelingHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keyword ``logit`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, M)`, where ``M = output_size``.

        Examples:
            >>> head = DuelingHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
        """
        a = self.A(x)
        v = self.V(x)
        q_value = a - a.mean(dim=-1, keepdim=True) + v
        return {'logit': q_value}


class StochasticDuelingHead(nn.Module):
    """
        Overview:
            The ``Stochastic Dueling Network`` proposed in paper ACER (arxiv 1611.01224). \
            Dueling network architecture in continuous action space. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            outputs ``q_value`` and ``v_value``.
        Interfaces:
            ``__init__``, ``forward``.
    """

    def __init__(
        self,
        hidden_size: int,
        action_shape: int,
        layer_num: int = 1,
        a_layer_num: Optional[int] = None,
        v_layer_num: Optional[int] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
        last_tanh: Optional[bool] = True,
    ) -> None:
        """
        Overview:
             Init the ``Stochastic DuelingHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``StochasticDuelingHead``.
            - action_shape (:obj:`int`): The number of continuous action shape, usually integer value.
            - layer_num (:obj:`int`): The number of default layers used in the network to compute action and value \
                output.
            - a_layer_num (:obj:`int`): The number of layers used in the network to compute action output. Default is \
                ``layer_num``.
            - v_layer_num (:obj:`int`): The number of layers used in the network to compute value output. Default is \
                ``layer_num``.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP. \
                Default ``False``.
            - last_tanh (:obj:`bool`): If ``True`` Apply ``tanh`` to actions. Default ``True``.
        """
        super(StochasticDuelingHead, self).__init__()
        if a_layer_num is None:
            a_layer_num = layer_num
        if v_layer_num is None:
            v_layer_num = layer_num
        layer = NoiseLinearLayer if noise else nn.Linear
        block = noise_block if noise else fc_block
        self.A = nn.Sequential(
            MLP(
                hidden_size + action_shape,
                hidden_size,
                hidden_size,
                a_layer_num,
                layer_fn=layer,
                activation=activation,
                norm_type=norm_type
            ), block(hidden_size, 1)
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
        if last_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

    def forward(
            self,
            s: torch.Tensor,
            a: torch.Tensor,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            sample_size: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Use encoded embedding tensor to run MLP with ``StochasticDuelingHead`` and return the prediction dictionary.
        Arguments:
            - s (:obj:`torch.Tensor`): Tensor containing input embedding.
            - a (:obj:`torch.Tensor`): The original continuous behaviour action.
            - mu (:obj:`torch.Tensor`): The ``mu`` gaussian reparameterization output of actor head at current \
                timestep.
            - sigma (:obj:`torch.Tensor`): The ``sigma`` gaussian reparameterization output of actor head at \
                current timestep.
            - sample_size (:obj:`int`): The number of samples for continuous action when computing the Q value.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords \
                ``q_value`` (:obj:`torch.Tensor`) and ``v_value`` (:obj:`torch.Tensor`).
        Shapes:
            - s: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - a: :math:`(B, A)`, where ``A = action_size``.
            - mu: :math:`(B, A)`.
            - sigma: :math:`(B, A)`.
            - q_value: :math:`(B, 1)`.
            - v_value: :math:`(B, 1)`.
        """

        batch_size = s.shape[0]  # batch_size or batch_size * T
        hidden_size = s.shape[1]
        action_size = a.shape[1]
        state_cat_action = torch.cat((s, a), dim=1)  # size (B, action_size + state_size)
        a_value = self.A(state_cat_action)  # size (B, 1)
        v_value = self.V(s)  # size (B, 1)
        # size (B, sample_size, hidden_size)
        expand_s = (torch.unsqueeze(s, 1)).expand((batch_size, sample_size, hidden_size))

        # in case for gradient back propagation
        dist = Independent(Normal(mu, sigma), 1)
        action_sample = dist.rsample(sample_shape=(sample_size, ))
        if self.tanh:
            action_sample = self.tanh(action_sample)
        # (sample_size, B, action_size)->(B, sample_size, action_size)
        action_sample = action_sample.permute(1, 0, 2)

        # size (B, sample_size, action_size + hidden_size)
        state_cat_action_sample = torch.cat((expand_s, action_sample), dim=-1)
        a_val_sample = self.A(state_cat_action_sample)  # size (B, sample_size, 1)
        q_value = v_value + a_value - a_val_sample.mean(dim=1)  # size (B, 1)

        return {'q_value': q_value, 'v_value': v_value}


class RegressionHead(nn.Module):
    """
        Overview:
            The ``RegressionHead`` used to output actions Q-value.
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``pred``.
        Interfaces:
            ``__init__``, ``forward``.
    """

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            layer_num: int = 2,
            final_tanh: Optional[bool] = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the ``RegressionHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``RegressionHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value output.
            - final_tanh (:obj:`bool`): If ``True`` apply ``tanh`` to output. Default ``False``.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
        """
        super(RegressionHead, self).__init__()
        self.main = MLP(hidden_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.last = nn.Linear(hidden_size, output_size)  # for convenience of special initialization
        self.final_tanh = final_tanh
        if self.final_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            Use encoded embedding tensor to run MLP with ``RegressionHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keyword ``pred`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - pred: :math:`(B, M)`, where ``M = output_size``.

        Examples:
            >>> head = RegressionHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['pred'].shape == torch.Size([4, 64])
        """
        x = self.main(x)
        x = self.last(x)
        if self.final_tanh:
            x = self.tanh(x)
        if x.shape[-1] == 1 and len(x.shape) > 1:
            x = x.squeeze(-1)
        return {'pred': x}


class ReparameterizationHead(nn.Module):
    """
        Overview:
            The ``ReparameterizationHead`` used to output action ``mu`` and ``sigma``.
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            outputs ``mu`` and ``sigma``.
        Interfaces:
            ``__init__``, ``forward``.
    """

    default_sigma_type = ['fixed', 'independent', 'conditioned']
    default_bound_type = ['tanh', None]

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 2,
        sigma_type: Optional[str] = None,
        fixed_sigma_value: Optional[float] = 1.0,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        bound_type: Optional[str] = None,
    ) -> None:
        """
        Overview:
            Init the ``ReparameterizationHead`` layers according to the provided arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to ``ReparameterizationHead``.
            - output_size (:obj:`int`): The number of outputs.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Q value output.
            - sigma_type (:obj:`str`): Sigma type used. Choose among \
                ``['fixed', 'independent', 'conditioned']``. Default is ``None``.
            - fixed_sigma_value (:obj:`float`): When choosing ``fixed`` type, the tensor ``output['sigma']`` \
                is filled with this input value. Default is ``None``.
            - activation (:obj:`nn.Module`): The type of activation function to use in MLP. \
                If ``None``, then default set activation to ``nn.ReLU()``. Default ``None``.
            - norm_type (:obj:`str`): The type of normalization to use. See ``ding.torch_utils.network.fc_block`` \
                for more details. Default ``None``.
            - bound_type (:obj:`str`): Bound type to apply to output ``mu``. Choose among ``['tanh', None]``. \
                Default is ``None``.
        """
        super(ReparameterizationHead, self).__init__()
        self.sigma_type = sigma_type
        assert sigma_type in self.default_sigma_type, "Please indicate sigma_type as one of {}".format(
            self.default_sigma_type
        )
        self.bound_type = bound_type
        assert bound_type in self.default_bound_type, "Please indicate bound_type as one of {}".format(
            self.default_bound_type
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
        Overview:
            Use encoded embedding tensor to run MLP with ``ReparameterizationHead`` and return the prediction \
            dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``mu`` (:obj:`torch.Tensor`) and ``sigma`` \
                (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - mu: :math:`(B, M)`, where ``M = output_size``.
            - sigma: :math:`(B, M)`.

        Examples:
            >>> head =  ReparameterizationHead(64, 64, sigma_type='fixed')
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['mu'].shape == torch.Size([4, 64])
            >>> assert outputs['sigma'].shape == torch.Size([4, 64])
        """
        x = self.main(x)
        mu = self.mu(x)
        if self.bound_type == 'tanh':
            mu = torch.tanh(mu)
        if self.sigma_type == 'fixed':
            sigma = self.sigma.to(mu.device) + torch.zeros_like(mu)  # addition aims to broadcast shape
        elif self.sigma_type == 'independent':
            log_sigma = self.log_sigma_param + torch.zeros_like(mu)  # addition aims to broadcast shape
            sigma = torch.exp(log_sigma)
        elif self.sigma_type == 'conditioned':
            log_sigma = self.log_sigma_layer(x)
            sigma = torch.exp(torch.clamp(log_sigma, -20, 2))
        return {'mu': mu, 'sigma': sigma}


class MultiHead(nn.Module):
    """
        Overview:
            The ``MultiHead`` used to output actions logit. \
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``logit``.
        Interfaces:
            ``__init__``, ``forward``.
    """

    def __init__(self, head_cls: type, hidden_size: int, output_size_list: SequenceType, **head_kwargs) -> None:
        """
        Overview:
            Init the ``MultiHead`` layers according to the provided arguments.
        Arguments:
            - head_cls (:obj:`type`): The class of head, choose among [``DuelingHead``, ``DistributionHead``, \
                ''QuatileHead'', ...].
            - hidden_size (:obj:`int`): The ``hidden_size`` of the MLP connected to the ``Head``.
            - output_size_list (:obj:`int`): Sequence of ``output_size`` for multi discrete action, e.g. ``[2, 3, 5]``.
            - head_kwargs: (:obj:`dict`): Dict containing class-specific arguments.
        """
        super(MultiHead, self).__init__()
        self.pred = nn.ModuleList()
        for size in output_size_list:
            self.pred.append(head_cls(hidden_size, size, **head_kwargs))

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            Use encoded embedding tensor to run MLP with ``MultiHead`` and return the prediction dictionary.
        Arguments:
            - x (:obj:`torch.Tensor`): Tensor containing input embedding.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``logit`` (:obj:`torch.Tensor`) \
                corresponding to the logit of each ``output`` each accessed at ``['logit'][i]``.
        Shapes:
            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.
            - logit: :math:`(B, Mi)`, where ``Mi = output_size`` corresponding to output ``i``.

        Examples:
            >>> head = MultiHead(DuelingHead, 64, [2, 3, 5], v_layer_num=2)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict)
            >>> # output_size_list is [2, 3, 5] as set
            >>> # Therefore each dim of logit is as follows
            >>> outputs['logit'][0].shape
            >>> torch.Size([4, 2])
            >>> outputs['logit'][1].shape
            >>> torch.Size([4, 3])
            >>> outputs['logit'][2].shape
            >>> torch.Size([4, 5])
        """
        return lists_to_dicts([m(x) for m in self.pred])


head_cls_map = {
    # discrete
    'discrete': DiscreteHead,
    'dueling': DuelingHead,
    'sdn': StochasticDuelingHead,
    'distribution': DistributionHead,
    'rainbow': RainbowHead,
    'qrdqn': QRDQNHead,
    'quantile': QuantileHead,
    # continuous
    'regression': RegressionHead,
    'reparameterization': ReparameterizationHead,
    # multi
    'multi': MultiHead,
}

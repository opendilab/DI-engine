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
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``DiscreteHead``
            - output_size (:obj:`int`): The number of output
            - layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
            - noise (:obj:`bool`): Whether use ``NoiseLinearLayer`` as ``layer_fn`` in Q networks' MLP
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
        r"""
        Overview:
            Use encoded embedding tensor to predict discrete output.
            Parameter updates with DiscreteHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``DiscreteHead`` setups
                and return the result prediction dictionary.

                Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.

        Examples:
            >>> head = DiscreteHead(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = head(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 64])
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
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``DistributionHead``
            - output_size (:obj:`int`): The num of output
            - layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
            - noise (:obj:`bool`): Whether use noisy ``fc_block``
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
        r"""
        Overview:
            Use encoded embedding tensor to predict Distribution output.
            Parameter updates with DistributionHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``DistributionHead`` setups and return the result prediction dictionary.

                Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
                    - distribution (:obj:`torch.Tensor`): Distribution tensor of size ``(B, N, n_atom)``
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
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``RainbowHead``
            - output_size (:obj:`int`): The num of output
            - layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
            - noise (:obj:`bool`): Whether use noisy ``fc_block``
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
        r"""
        Overview:
            Use encoded embedding tensor to predict Rainbow output.
            Parameter updates with RainbowHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``RainbowHead`` setups and return the result prediction dictionary.

                Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
                    - distribution (:obj:`torch.Tensor`): Distribution tensor of size ``(B, N, n_atom)``
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
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``QRDQNHead``
            - output_size (:obj:`int`): The num of output
            - layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
            - noise (:obj:`bool`): Whether use noisy ``fc_block``
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
        r"""
        Overview:
            Use encoded embedding tensor to predict QRDQN output.
            Parameter updates with QRDQNHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``QRDQNHead`` setups and return the result prediction dictionary.

                Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
                    - q (:obj:`torch.Tensor`): Q valye tensor tensor of size ``(B, N, num_quantiles)``
                    - tau (:obj:`torch.Tensor`): tau tensor of size ``(B, N, 1)``
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
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``QuantileHead``
            - output_size (:obj:`int`): The num of output
            - layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
            - noise (:obj:`bool`): Whether use noisy ``fc_block``
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
        r"""
        Overview:
           Deterministic parametric function trained to reparameterize samples from a base distribution.
           By repeated Bellman update iterations of Q-learning, the optimal action-value function is estimated.
        Arguments:
            - x (:obj:`torch.Tensor`): The encoded embedding tensor of parametric sample
        Returns:
            - (:obj:`torch.Tensor`):
                QN output tensor after reparameterization of shape ``(quantile_embedding_size, output_size)``
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
        r"""
        Overview:
            Use encoded embedding tensor to predict Quantile output.
            Parameter updates with QuantileHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``QuantileHead`` setups and return the result prediction dictionary.

                Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
                    - q (:obj:`torch.Tensor`): Q valye tensor tensor of size ``(num_quantiles, B, N)``
                    - quantiles (:obj:`torch.Tensor`): quantiles tensor of size ``(quantile_embedding_size, 1)``
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
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``DuelingHead``
            - output_size (:obj:`int`): The num of output
            - a_layer_num (:obj:`int`): The num of layers used in the network to compute action output
            - v_layer_num (:obj:`int`): The num of layers used in the network to compute value output
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
            - noise (:obj:`bool`): Whether use noisy ``fc_block``
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
        r"""
        Overview:
            Use encoded embedding tensor to predict Dueling output.
            Parameter updates with DuelingHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``DuelingHead`` setups and return the result prediction dictionary.

                Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
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
            The Stochastic Dueling Network proposed in paper ACER (arxiv 1611.01224), dueling netwowrk architecture in \
            continuous action space. Initialize the head according to input arguments.
        Arguments:
            - hidden_size (:obj:`int`): The num of observation embedding size.
            - action_shape (:obj:`int`): The num of continuous action shape, usually integer value.
            - a_layer_num (:obj:`int`): The num of layers used in the network to compute action output.
            - v_layer_num (:obj:`int`): The num of layers used in the network to compute value output.
            - activation (:obj:`nn.Module`): The type of activation function to use in ``MLP`` after ``layer_fn``, \
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`): The type of normalization to use, see ``ding.torch_utils.fc_block`` for \
                more details.
            - noise (:obj:`bool`): Whether to use noisy ``fc_block`` for more exploration.
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
            Use encoded observation, behaviour action and sampled actions with (mu, sigma) output by actor head \
            at current timestep to get dueling Q-value, i.e. continuous dueling head.
        Arguments:
            - s (:obj:`torch.Tensor`): The encoded embedding state tensor, determined with given ``hidden_size``, \
                i.e. shape is ``(B, N=hidden_size)``.
            - a (:obj:`torch.Tensor`): The original continuous behaviour action, determined with ``action_size`` \
                i.e. shape is ``(B, N=action_size)``.
            - mu (:obj:`torch.Tensor`):
                The mu gaussian reparameterization output of actor head at current timestep, size (B, action_size)
            - sigma (:obj:`torch.Tensor`):
                The sigma gaussian reparameterization output of actor head at current timestep, size (B, action_size)
            - sample_size (:obj:`int`): The number of samples for continuous action when computing the Q value
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): Output dict data, including q_value and v_value tensor, \
                and their shape is ``(B, 1)``.
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

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            layer_num: int = 2,
            final_tanh: Optional[bool] = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``RegressionHead``
            - output_size (:obj:`int`): The num of output
            - final_tanh (:obj:`Optional[bool]`): Whether a final tanh layer is needed
            - layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
        """
        super(RegressionHead, self).__init__()
        self.main = MLP(hidden_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.last = nn.Linear(hidden_size, output_size)  # for convenience of special initialization
        self.final_tanh = final_tanh
        if self.final_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict Regression output.
            Parameter updates with RegressionHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``RegressionHead`` setups and return the result prediction dictionary.

                Necessary Keys:
                     - pred (:obj:`torch.Tensor`): Tensor with prediction value cells, with same size as input ``x``.
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
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - hidden_size (:obj:`int`): The ``hidden_size`` used before connected to ``ReparameterizationHead``
            - output_size (:obj:`int`): The num of output
            - layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - sigma_type (:obj:`Optional[str]`): Sigma type used in ``['fixed', 'independent', 'conditioned']``
            - fixed_sigma_value(:obj:`Optional[float]`):
                When choosing ``fixed`` type, the tensor ``output['sigma']`` is filled with this input value.
            - activation (:obj:`nn.Module`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details
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
        r"""
        Overview:
            Use encoded embedding tensor to predict Reparameterization output.
            Parameter updates with ReparameterizationHead's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``ReparameterizationHead`` setups and return the result prediction dictionary.

                Necessary Keys:
                    - mu (:obj:`torch.Tensor`) Tensor of cells of updated mu values of size ``(B, action_size)``
                    - sigma (:obj:`torch.Tensor`) Tensor of cells of updated sigma values of size ``(B, action_size)``
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

    def __init__(self, head_cls: type, hidden_size: int, output_size_list: SequenceType, **head_kwargs) -> None:
        r"""
        Overview:
            Init the MultiHead according to arguments.
        Arguments:
            - head_cls (:obj:`type`):
                The class of head, like ``DuelingHead``, ``DistributionHead``, ``QuatileHead``, etc
            - hidden_size (:obj:`int`): The number of hidden layer size
            - output_size_list (:obj:`int`):
                The collection of ``output_size``, e.g.: multi discrete action, ``[2, 3, 5]``
            - head_kwargs: (:obj:`dict`): Class-specific arguments
        """
        super(MultiHead, self).__init__()
        self.pred = nn.ModuleList()
        for size in output_size_list:
            self.pred.append(head_cls(hidden_size, size, **head_kwargs))

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict multi discrete output
        Arguments:
            - x (:obj:`torch.Tensor`): The encoded embedding tensor, usually with shape ``(B, N)``
        Returns:
            - outputs (:obj:`Dict`):
                Prediction output dict

                Necessary Keys:
                    - logit (:obj:`torch.Tensor`):
                        Logit tensor with logit tensors indexed by ``output`` each accessed at ``['logit'][i]``.
                        Given that ``output_size_list==[o1,o2,o3,...]`` , ``['logit'][i]`` is of size ``(B,Ni)``

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

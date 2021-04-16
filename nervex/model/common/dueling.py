from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nervex.torch_utils import fc_block, noise_block
from typing import Callable
from nervex.rl_utils import beta_function_map


class DuelingHead(nn.Module):
    r"""
    Overview:
        The Dueling head used in models.
    Interfaces:
        __init__, forward

    .. note::
        Dueling head is one of the three most major improvements in DQN algorithm. The paper introducing \
            this improvement `Dueling Network Architectures for Deep Reinforcement Learning` was published \
            by Google in 2016. You can view the original paper on <https://arxiv.org/pdf/1511.06581.pdf>
        The other two major improvements are double DQN and prioritized replay, which in nerveX \
            are implemented though plugins and buffer.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        a_layer_num: int,
        v_layer_num: int,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        init_type: Optional[str] = "xavier",
        distribution: bool = False,
        quantile: bool = False,
        noise: bool = False,
        v_min: float = -10,
        v_max: float = 10,
        n_atom: int = 51,
        num_quantiles: int = 32,
        quantile_embedding_dim: int = 128,
        beta_function_type: str = 'uniform',
    ) -> None:
        r"""
        Overview:
            Init the DuelingHead according to arguments.
        Arguments:
            - hidden_dim (:obj:`int`): the hidden_dim used before connected to DuelingHead
            - action_dim (:obj:`int`): the num of actions
            - a_layer_num (:obj:`int`): the num of fc_block used in the network to compute action output
            - v_layer_num (:obj:`int`): the num of fc_block used in the network to compute value output
            - activation (:obj:`nn.Module`): the type of activation to use in the fc_block,\
                if None then default set to nn.ReLU
            - norm_type (:obj:`str`): the type of normalization to use, see nervex.torch_utils.fc_block for more details
            - distribution (:obj:`bool`): whether output q_value form distribution perspective
            - noise (:obj:`bool`): whether use noisy fc block
            - v_min (:obj:`float`): value distribution minimum value
            - v_max (:obj:`float`): value distribution maximum value
            - n_atom (:obj:`int`): the number of atom sample point
        """
        super(DuelingHead, self).__init__()
        self.noise = noise
        self.distribution = distribution
        self.quantile = quantile
        self.num_quantiles = num_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim
        self.action_dim = action_dim
        self.v_min = v_min
        self.v_max = v_max
        self.n_atom = n_atom
        if self.quantile:
            self.iqn_fc = nn.Linear(self.quantile_embedding_dim, hidden_dim)
            self.beta_function = beta_function_map[beta_function_type]
        if noise:
            block = noise_block
        else:
            block = fc_block
        self.A = [
            block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type, init_type=init_type)
            for _ in range(a_layer_num)
        ]
        self.V = [
            block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type, init_type=init_type)
            for _ in range(v_layer_num)
        ]

        a_out_dim = action_dim
        v_out_dim = 1

        if self.distribution:
            a_out_dim *= n_atom
            v_out_dim *= n_atom

        self.A += block(hidden_dim, a_out_dim, activation=None, norm_type=None)
        self.V += block(hidden_dim, v_out_dim, activation=None, norm_type=None)

        self.A = nn.Sequential(*self.A)
        self.V = nn.Sequential(*self.V)

    def forward(self,
                x: torch.Tensor,
                num_quantiles: Union[int, None] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Overview:
            Return the sum of advantage and the value according to the input from hidden layers
        Arguments:
            - x (:obj:`torch.Tensor`): the input from hidden layers
        Returns:
            - return (:obj:`torch.Tensor`): the sum of advantage and value
        """
        batch_size = x.shape[0]
        device = torch.device("cuda" if x.is_cuda else "cpu")
        if self.quantile:
            if not num_quantiles:
                num_quantiles = self.num_quantiles

            quantiles = torch.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1).to(device)

            beta_quantiles = self.beta_function(quantiles)

            quantile_net = quantiles.repeat([1, self.quantile_embedding_dim])
            beta_quantile_net = beta_quantiles.repeat([1, self.quantile_embedding_dim])

            quantile_net = torch.cos(
                torch.arange(1, self.quantile_embedding_dim + 1, 1, device=device, dtype=torch.float32) * math.pi *
                quantile_net
            )
            quantile_net = self.iqn_fc(quantile_net)
            quantile_net = F.relu(quantile_net)

            beta_quantile_net = torch.cos(
                torch.arange(1, self.quantile_embedding_dim + 1, 1, device=device, dtype=torch.float32) * math.pi *
                beta_quantile_net
            )
            beta_quantile_net = self.iqn_fc(beta_quantile_net)
            beta_quantile_net = F.relu(beta_quantile_net)

            x = x.repeat(num_quantiles, 1)

            x = x * quantile_net
            beta_x = x * beta_quantile_net

        a = self.A(x)
        v = self.V(x)

        if self.distribution:
            a = a.view(*a.shape[:-1], self.action_dim, self.n_atom)
            v = v.view(*v.shape[:-1], 1, self.n_atom)
            dist = a - a.mean(dim=-2, keepdim=True) + v
            dist = torch.softmax(dist, dim=-1) + 1e-6
            q = dist * torch.linspace(self.v_min, self.v_max,
                                      self.n_atom).to(torch.device("cuda" if dist.is_cuda else "cpu"))
            q = q.sum(-1)
            return q, dist
        elif self.quantile:
            beta_a = self.A(beta_x)
            beta_v = self.V(beta_x)
            q = a - a.mean(dim=-1, keepdim=True) + v
            q = q.reshape(num_quantiles, batch_size, -1)
            beta_q = beta_a - beta_a.mean(dim=-1, keepdim=True) + beta_v
            beta_q = beta_q.reshape(num_quantiles, batch_size, -1)
            logit = beta_q.mean(0)
            return logit, q, quantiles
        else:
            return a - a.mean(dim=-1, keepdim=True) + v

from typing import Union, Optional

import torch
import torch.nn as nn

from nervex.torch_utils import fc_block, noise_block


class DuelingHead(nn.Module):
    r"""
    Overview:
        The Dueling head used in models.
    Notes:
        Dueling head is one of the three most major improvements in DQN algorithm. The paper introducing \
            this improvement `Dueling Network Architectures for Deep Reinforcement Learning` was published \
            by Google in 2016. You can view the original paper on <https://arxiv.org/pdf/1511.06581.pdf>
        The other two major improvements are double DQN and prioritized replay, which in nerveX \
            are implemented though plugins and buffer.
    Interfaces:
        __init__, forward
    """

    def __init__(
            self,
            hidden_dim: int,
            action_dim: int,
            a_layer_num: int,
            v_layer_num: int,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            distribution: bool = False,
            noise: bool = False,
            v_min: float = -10,
            v_max: float = 10,
            num_atom: int = 51,
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
        """
        super(DuelingHead, self).__init__()
        self.noise = noise
        self.distribution = distribution
        self.action_dim = action_dim
        self.num_atom = num_atom
        if noise:
            block = noise_block
        else:
            block = fc_block
        self.A = [
            block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type) for _ in range(a_layer_num)
        ]
        self.V = [
            block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type) for _ in range(v_layer_num)
        ]

        a_out_dim = action_dim
        v_out_dim = 1

        if self.distribution:
            a_out_dim *= num_atom
            v_out_dim *= num_atom

        self.A += block(hidden_dim, a_out_dim, activation=None, norm_type=None)
        self.V += block(hidden_dim, v_out_dim, activation=None, norm_type=None)

        self.A = nn.Sequential(*self.A)
        self.V = nn.Sequential(*self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the sum of advantage and the value according to the input from hidden layers
        Arguments:
            - x (:obj:`torch.Tensor`): the input from hidden layers
        Returns:
            - return (:obj:`torch.Tensor`): the sum of advantage and value
        """
        a = self.A(x)
        v = self.V(x)
        if self.distribution:
            a = a.view(-1, self.action_dim, self.num_atom)
            v = v.view(-1, 1, self.num_atom)
        return a - a.mean(dim=-1, keepdim=True) + v

from typing import Union

import torch
import torch.nn as nn

from nervex.torch_utils import fc_block


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
            activation: Union[None, nn.Module] = nn.ReLU(),
            norm_type: Union[None, str] = None
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
        self.A = [
            fc_block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type) for _ in range(a_layer_num)
        ]
        self.V = [
            fc_block(hidden_dim, hidden_dim, activation=activation, norm_type=norm_type) for _ in range(v_layer_num)
        ]

        self.A += fc_block(hidden_dim, action_dim, activation=None, norm_type=None)
        self.V += fc_block(hidden_dim, 1, activation=None, norm_type=None)

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
        return a - a.mean(dim=-1, keepdim=True) + v

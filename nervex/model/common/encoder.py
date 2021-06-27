from typing import Optional
import torch
import torch.nn as nn

from nervex.torch_utils import fc_block, ResFCBlock, ResBlock
from nervex.utils import SequenceType


class ConvEncoder(nn.Module):
    r"""
    Overview:
        The Convolution Encoder used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        __init__, forward
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [32, 64, 64],
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list

        layers = []
        kernel_size = [8, 4, 3]
        stride = [4, 2, 1]
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i]))
            layers.append(self.act)
            input_size = hidden_size_list[i]
        assert len(set(hidden_size_list[3:])) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(3, len(self.hidden_size_list)):
            layers.append(ResBlock(self.hidden_size_list[i], activation=self.act, norm_type=norm_type))
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.mid = nn.Linear(flatten_size, hidden_size_list[-1])

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main``
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): env raw observation
        Returns:
            - return (:obj:`torch.Tensor`): embedding tensor
        """
        x = self.main(x)
        x = self.mid(x)
        return x


class FCEncoder(nn.Module):

    def __init__(
            self,
            obs_shape: int,
            hidden_size_list: SequenceType,
            res_block: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])

        if res_block:
            assert len(set(hidden_size_list)) == 1, "Please indicate the same hidden size for res block parts"
            if len(hidden_size_list) == 1:
                self.main = ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type)
            else:
                layers = []
                for i in range(len(hidden_size_list)):
                    layers.append(ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type))
                self.main = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(len(hidden_size_list) - 1):
                layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
                layers.append(self.act)
            self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.init(x))
        x = self.main(x)
        return x


class StructEncoder(nn.Module):
    # TODO(nyz)
    pass

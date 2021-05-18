from typing import Optional
import torch
import torch.nn as nn

from nervex.torch_utils import fc_block, ResFCBlock


class ConvEncoder(nn.Module):
    r"""
    Overview:
        The Convolution Encoder used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        __init__, forward
    """

    def __init__(self, obs_shape: tuple, embedding_size: int) -> None:
        r"""
        Overview:
            Init the DuelingHead according to arguments.
        Arguments:
            - obs_shape (:obj:`tuple`): a tuple of observation shape
            - embedding_size (:obj:`int`): output size of this encoder
        """
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = nn.ReLU()
        layers = []
        hidden_size_list = [32, 64, 64]  # out_channels
        kernel_size = [8, 4, 3]
        stride = [4, 2, 1]
        input_size = obs_shape[0]  # in_channel
        for i in range(len(hidden_size_list)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i]))
            layers.append(self.act)
            input_size = hidden_size_list[i]
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)
        flatten_size = self._get_flatten_size()
        self.mid = nn.Linear(flatten_size, embedding_size)

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

    def __init__(self, obs_shape: int, embedding_size: int, norm_type: Optional[str] = None) -> None:
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = nn.ReLU()
        self.init = nn.Linear(obs_shape, embedding_size)
        self.main = ResFCBlock(embedding_size, activation=self.act, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.init(x))
        x = self.main(x)
        return x

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

    def __init__(self, obs_dim: tuple, embedding_dim: int) -> None:
        r"""
        Overview:
            Init the DuelingHead according to arguments.
        Arguments:
            - obs_dim (:obj:`tuple`): a tuple of observation dim
            - embedding_dim (:obj:`int`): output dim of this encoder
        """
        super(ConvEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.act = nn.ReLU()
        layers = []
        hidden_dim_list = [32, 64, 64]  # out_channels
        kernel_size = [8, 4, 3]
        stride = [4, 2, 1]
        input_dim = obs_dim[0]  # in_channel
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, hidden_dim_list[i], kernel_size[i], stride[i]))
            layers.append(self.act)
            input_dim = hidden_dim_list[i]
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)
        flatten_dim = self._get_flatten_dim()
        self.mid = nn.Linear(flatten_dim, embedding_dim)

    def _get_flatten_dim(self) -> int:
        r"""
        Overview:
            Get the encoding dim after ``self.main``
        """
        test_data = torch.randn(1, *self.obs_dim)
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

    def __init__(self, obs_dim: int, embedding_dim: int) -> None:
        super(FCEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.act = nn.ReLU()
        self.main = ResFCBlock(obs_dim, activation=self.act)
        self.mid = nn.Linear(obs_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        x = self.mid(x)
        return x

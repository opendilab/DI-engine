import torch
import torch.nn as nn

from nervex.torch_utils import fc_block, ResFCBlock


class ConvEncoder(nn.Module):

    def __init__(self, obs_dim: tuple, embedding_dim: int) -> None:
        super(ConvEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.act = nn.ReLU()
        layers = []
        hidden_dim_list = [32, 64, 64]
        kernel_size = [8, 4, 3]
        stride = [4, 2, 1]
        input_dim = obs_dim[0]
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, hidden_dim_list[i], kernel_size[i], stride[i]))
            layers.append(self.act)
            input_dim = hidden_dim_list[i]
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)
        flatten_dim = self._get_flatten_dim()
        self.mid = nn.Linear(flatten_dim, embedding_dim)

    def _get_flatten_dim(self) -> int:
        test_data = torch.randn(1, *self.obs_dim)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

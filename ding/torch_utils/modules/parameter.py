import torch
from torch import nn
from torch.distributions.transforms import TanhTransform


class NonegativeParameter(nn.Module):

    def __init__(self, data=None, requires_grad=True, delta=1e-8):
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.log_data = nn.Parameter(torch.log(data + delta), requires_grad=requires_grad)

    def forward(self):
        return torch.exp(self.log_data)

    @property
    def data(self):
        return torch.exp(self.log_data)


class TanhParameter(nn.Module):

    def __init__(self, data=None, requires_grad=True):
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.transform = TanhTransform(cache_size=1)

        self.data_inv = nn.Parameter(self.transform.inv(data), requires_grad=requires_grad)

    def forward(self):
        return self.transform(self.data_inv)

    @property
    def data(self):
        return self.transform(self.data_inv)

from typing import Optional, Union

import math
import torch
import torch.nn as nn


class PopArt(nn.Module):
    r"""
    Overview:
        A linear layer with popart normalization.
    Interfaces:
        forward
    .. note:

        For more popart info, you can refer to
        the paper <https://arxiv.org/abs/1809.04474>
    """

    def __init__(
            self,
            input_features: Union[int, None] = None,
            output_features: Union[int, None] = None,
            beta: float = 0.5
    ) -> None:

        super(PopArt, self).__init__()

        self.beta = beta
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))

        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))

        self.mean = torch.zeros(output_features, requires_grad=False)
        self.v = torch.ones(output_features, requires_grad=False)
        self.std = torch.ones(output_features, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
            Overview:
                Return the popart output.
            Arguments:
                - x (:obj:`torch.Tensor`): The input tensor.
            Returns:
                - x (:obj:`torch.Tensor`): The output tensor.
            """

        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)

        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return normalized_output

    def update_parameters(self, value):
        r"""
            Overview:
                Update the popart parameters.
            Arguments:
                - value (:obj:`torch.Tensor`): The input q value tensor.
            Returns:
                - output (:obj:`Dict`): The dict of popart normalization parameters.
            """

        self.mean = self.mean.to(value.device)
        self.std = self.mean.to(value.device)
        self.v = self.mean.to(value.device)

        old_mu = self.mu
        old_std = self.std
        old_v = self.v

        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)

        batch_mean[torch.isnan(batch_mean)] = self.mean[torch.isnan(batch_mean)]
        batch_v[torch.isnan(batch_v)] = self.v[torch.isnan(batch_v)]

        batch_mean = (1 - self.beta) * self.mu + self.beta * batch_mean
        batch_v = (1 - self.beta) * self.v + self.beta * batch_v

        batch_std = torch.sqrt(batch_v - (batch_mean ** 2))
        batch_std = torch.clamp(batch_std, min=1e-4, max=1e+6)
        batch_std[torch.isnan(batch_std)] = self.std[torch.isnan(batch_std)]

        self.mu = batch_mean
        self.v = batch_v
        self.std = batch_std

        self.weight.data = (self.weight.t() * old_std / self.std).t()
        self.bias.data = (old_std * self.bias + old_mu - self.mu) / self.std

        return {'new_mean': batch_mean, 'new_std': batch_std}

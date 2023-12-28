"""
Implementation of ``POPART`` algorithm for reward rescale.
<link https://arxiv.org/abs/1602.07714 link>

POPART is an adaptive normalization algorithm to normalize the targets used in the learning updates.
The two main components in POPART are:
**ART**: to update scale and shift such that the return is appropriately normalized,
**POP**: to preserve the outputs of the unnormalized function when we change the scale and shift.

"""
from typing import Optional, Union, Dict
import math
import torch
import torch.nn as nn


class PopArt(nn.Module):
    """
    Overview:
        A linear layer with popart normalization. This class implements a linear transformation followed by
        PopArt normalization, which is a method to automatically adapt the contribution of each task to the agent's
        updates in multi-task learning, as described in the paper <https://arxiv.org/abs/1809.04474>.

    Interface:
        __init__, reset_parameters, forward, update_parameters
    """

    def __init__(
            self,
            input_features: Union[int, None] = None,
            output_features: Union[int, None] = None,
            beta: float = 0.5
    ) -> None:
        """
        Overview:
            Initialize the class with input features, output features, and the beta parameter.
        Arguments:
            - input_features (:obj:`Union[int, None]`): The size of each input sample.
            - output_features (:obj:`Union[int, None]`): The size of each output sample.
            - beta (:obj:`float`): The parameter for moving average.
        """
        super(PopArt, self).__init__()

        self.beta = beta
        self.input_features = input_features
        self.output_features = output_features
        # Initialize the linear layer parameters, weight and bias.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        # Register a buffer for normalization parameters which can not be considered as model parameters.
        # The normalization parameters will be used later to save the target value's scale and shift.
        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))
        self.register_buffer('v', torch.ones(output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Overview:
            Reset the parameters including weights and bias using kaiming_uniform_ and uniform_ initialization.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Implement the forward computation of the linear layer and return both the output and the
            normalized output of the layer.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor which is to be normalized.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): A dictionary contains 'pred' and 'unnormalized_pred'.
        """
        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)
        # The unnormalization of output
        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return {'pred': normalized_output.squeeze(1), 'unnormalized_pred': output.squeeze(1)}

    def update_parameters(self, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Update the normalization parameters based on the given value and return the new mean and
            standard deviation after the update.
        Arguments:
            - value (:obj:`torch.Tensor`): The tensor to be used for updating parameters.
        Returns:
            - update_results (:obj:`Dict[str, torch.Tensor]`): A dictionary contains 'new_mean' and 'new_std'.
        """
        # Tensor device conversion of the normalization parameters.
        self.mu = self.mu.to(value.device)
        self.sigma = self.sigma.to(value.device)
        self.v = self.v.to(value.device)

        old_mu = self.mu
        old_std = self.sigma

        # Calculate the first and second moments (mean and variance) of the target value:
        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)
        batch_mean[torch.isnan(batch_mean)] = self.mu[torch.isnan(batch_mean)]
        batch_v[torch.isnan(batch_v)] = self.v[torch.isnan(batch_v)]
        batch_mean = (1 - self.beta) * self.mu + self.beta * batch_mean
        batch_v = (1 - self.beta) * self.v + self.beta * batch_v
        batch_std = torch.sqrt(batch_v - (batch_mean ** 2))
        # Clip the standard deviation to reject the outlier data.
        batch_std = torch.clamp(batch_std, min=1e-4, max=1e+6)
        # Replace the nan value with old value.
        batch_std[torch.isnan(batch_std)] = self.sigma[torch.isnan(batch_std)]

        self.mu = batch_mean
        self.v = batch_v
        self.sigma = batch_std
        # Update weight and bias with mean and standard deviation to preserve unnormalised outputs
        self.weight.data = (self.weight.data.t() * old_std / self.sigma).t()
        self.bias.data = (old_std * self.bias.data + old_mu - self.mu) / self.sigma

        return {'new_mean': batch_mean, 'new_std': batch_std}

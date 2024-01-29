from typing import Optional
import torch
from torch import nn
from torch.distributions.transforms import TanhTransform


class NonegativeParameter(nn.Module):
    """
    Overview:
        This module will output a non-negative parameter during the forward process.
    Interfaces:
        ``__init__``, ``forward``, ``set_data``.
    """

    def __init__(self, data: Optional[torch.Tensor] = None, requires_grad: bool = True, delta: float = 1e-8):
        """
        Overview:
            Initialize the NonegativeParameter object using the given arguments.
        Arguments:
            - data (:obj:`Optional[torch.Tensor]`): The initial value of generated parameter. If set to ``None``, the \
                default value is 0.
            - requires_grad (:obj:`bool`): Whether this parameter requires grad.
            - delta (:obj:`Any`): The delta of log function.
        """
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.log_data = nn.Parameter(torch.log(data + delta), requires_grad=requires_grad)

    def forward(self) -> torch.Tensor:
        """
        Overview:
            Output the non-negative parameter during the forward process.
        Returns:
            parameter (:obj:`torch.Tensor`): The generated parameter.
        """
        return torch.exp(self.log_data)

    def set_data(self, data: torch.Tensor) -> None:
        """
        Overview：
            Set the value of the non-negative parameter.
        Arguments:
            data (:obj:`torch.Tensor`): The new value of the non-negative parameter.
        """
        self.log_data = nn.Parameter(torch.log(data + 1e-8), requires_grad=self.log_data.requires_grad)


class TanhParameter(nn.Module):
    """
    Overview:
        This module will output a tanh parameter during the forward process.
    Interfaces:
        ``__init__``, ``forward``, ``set_data``.
    """

    def __init__(self, data: Optional[torch.Tensor] = None, requires_grad: bool = True):
        """
        Overview:
            Initialize the TanhParameter object using the given arguments.
        Arguments:
            - data (:obj:`Optional[torch.Tensor]`): The initial value of generated parameter. If set to ``None``, the \
                default value is 1.
            - requires_grad (:obj:`bool`): Whether this parameter requires grad.
        """
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.transform = TanhTransform(cache_size=1)

        self.data_inv = nn.Parameter(self.transform.inv(data), requires_grad=requires_grad)

    def forward(self) -> torch.Tensor:
        """
        Overview:
            Output the tanh parameter during the forward process.
        Returns:
            parameter (:obj:`torch.Tensor`): The generated parameter.
        """
        return self.transform(self.data_inv)

    def set_data(self, data: torch.Tensor) -> None:
        """
        Overview:
            Set the value of the tanh parameter.
        Arguments:
            data (:obj:`torch.Tensor`): The new value of the tanh parameter.
        """
        self.data_inv = nn.Parameter(self.transform.inv(data), requires_grad=self.data_inv.requires_grad)

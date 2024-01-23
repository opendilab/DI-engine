import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSoftmax(nn.Module):
    """
    Overview:
        An `nn.Module` that computes GumbelSoftmax.
    Interfaces:
        ``__init__``, ``forward``, ``gumbel_softmax_sample``

    .. note::
        For more information on GumbelSoftmax, refer to the paper [Categorical Reparameterization \
        with Gumbel-Softmax](https://arxiv.org/abs/1611.01144).
    """

    def __init__(self) -> None:
        """
         Overview:
             Initialize the `GumbelSoftmax` module.
         """
        super(GumbelSoftmax, self).__init__()

    def gumbel_softmax_sample(self, x: torch.Tensor, temperature: float, eps: float = 1e-8) -> torch.Tensor:
        """
        Overview:
            Draw a sample from the Gumbel-Softmax distribution.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
            - temperature (:obj:`float`): Non-negative scalar controlling the sharpness of the distribution.
            - eps (:obj:`float`): Small number to prevent division by zero, default is `1e-8`.
        Returns:
            - output (:obj:`torch.Tensor`): Sample from Gumbel-Softmax distribution.
        """
        U = torch.rand(x.shape)
        U = U.to(x.device)
        y = x - torch.log(-torch.log(U + eps) + eps)
        return F.softmax(y / temperature, dim=1)

    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Overview:
            Forward pass for the `GumbelSoftmax` module.
        Arguments:
            - x (:obj:`torch.Tensor`): Unnormalized log-probabilities.
            - temperature (:obj:`float`): Non-negative scalar controlling the sharpness of the distribution.
            - hard (:obj:`bool`): If `True`, returns one-hot encoded labels. Default is `False`.
        Returns:
            - output (:obj:`torch.Tensor`): Sample from Gumbel-Softmax distribution.
        Shapes:
            - x: its shape is :math:`(B, N)`, where `B` is the batch size and `N` is the number of classes.
            - y: its shape is :math:`(B, N)`, where `B` is the batch size and `N` is the number of classes.
        """
        y = self.gumbel_softmax_sample(x, temperature)
        if hard:
            y_hard = torch.zeros_like(x)
            y_hard[torch.arange(0, x.shape[0]), y.max(1)[1]] = 1
            # The detach function treat (y_hard - y) as constant,
            # to make sure makes the gradient equal to y_soft gradient
            y = (y_hard - y).detach() + y
        return y

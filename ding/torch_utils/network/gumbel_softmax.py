import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSoftmax(nn.Module):
    r"""
    Overview:
        An nn.Module that computes GumbelSoftmax
    Interface:
        __init__, forward

    .. note:

        For more gumbelsoftmax info, you can refer to
        the paper <https://arxiv.org/abs/1611.01144>

    """

    def __init__(self) -> None:
        r"""
        Overview:
            Initialize the GumbelSoftmax module
        """
        super(GumbelSoftmax, self).__init__()

    def gumbel_softmax_sample(self, x: torch.Tensor, temperature, eps=1e-8):
        """ Draw a sample from GumbelSoftmax distribution"""
        U = torch.rand(x.shape)
        U = U.to(x.device)
        y = x - torch.log(-torch.log(U + eps) + eps)
        return F.softmax(y / temperature, dim=1)

    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        r"""
        Arguments:
            - x (:obj:`torch.Tensor`): unnormalized log-probs
            - temperature(:obj:`float`): non-negative scalar
            - hard(:obj:`bool`): if true return one-hot label
        Returns:
            - output (:obj:`torch.Tensor`): sample from GumbelSoftmax distribution
        Shapes:
            - x: :math:`(B, N)`, while B is the batch size, N is number of classes
            - output: :math:`(B, N)`, while B is the batch size, N is number of classes
        """
        y = self.gumbel_softmax_sample(x, temperature)
        if hard:
            y_hard = torch.zeros_like(x)
            y_hard[torch.arange(0, x.shape[0]), y.max(1)[1]] = 1
            # The detach function treat (y_hard - y) as constant,
            # to make sure makes the gradient equal to y_soft gradient
            y = (y_hard - y).detach() + y
        return y

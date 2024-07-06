import torch


def value_transform(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Overview:
        A function to reduce the scale of the action-value function.
        :math: `h(x) = sign(x)(\sqrt{(abs(x)+1)} - 1) + \epsilon * x` .
    Arguments:
        - x: (:obj:`torch.Tensor`) The input tensor to be normalized.
        - eps: (:obj:`float`) The coefficient of the additive regularization term \
            to ensure inverse function is Lipschitz continuous
    Returns:
        - (:obj:`torch.Tensor`) Normalized tensor.

    .. note::
        Observe and Look Further: Achieving Consistent Performance on Atari (https://arxiv.org/abs/1805.11593).
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def value_inv_transform(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Overview:
        The inverse form of value rescale.
        :math: `h^{-1}(x) = sign(x)({(\frac{\sqrt{1+4\epsilon(|x|+1+\epsilon)}-1}{2\epsilon})}^2-1)` .
    Arguments:
        - x: (:obj:`torch.Tensor`) The input tensor to be unnormalized.
        - eps: (:obj:`float`) The coefficient of the additive regularization term \
            to ensure inverse function is Lipschitz continuous
    Returns:
        - (:obj:`torch.Tensor`) Unnormalized tensor.
    """
    return torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        A function to normalize the targets.
        :math: `symlog(x) = sign(x)(\ln{|x|+1})` .
    Arguments:
        - x: (:obj:`torch.Tensor`) The input tensor to be normalized.
    Returns:
        - (:obj:`torch.Tensor`) Normalized tensor.

    .. note::
        Mastering Diverse Domains through World Models (https://arxiv.org/abs/2301.04104)
    """
    return torch.sign(x) * (torch.log(torch.abs(x) + 1))


def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        The inverse form of symlog.
        :math: `symexp(x) = sign(x)(\exp{|x|}-1)` .
    Arguments:
        - x: (:obj:`torch.Tensor`) The input tensor to be unnormalized.
    Returns:
        - (:obj:`torch.Tensor`) Unnormalized tensor.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

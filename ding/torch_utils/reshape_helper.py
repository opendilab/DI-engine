from typing import Tuple, Union

from torch import Tensor, Size


def fold_batch(x: Tensor, nonbatch_ndims: int = 1) -> Tuple[Tensor, Size]:
    r"""
    Overview:
        :math:`(T, B, X) \leftarrow (T*B, X)`\
        Fold the first (ndim - nonbatch_ndims) dimensions of a tensor as batch dimension.\
        This operation is similar to `torch.flatten` but provides an inverse function
        `unfold_batch` to restore the folded dimensions.

    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to fold
        - nonbatch_ndims (:obj:`int`): the number of dimensions that is not folded as
            batch dimension.

    Returns:
        - x (:obj:`torch.Tensor`): the folded tensor
        - batch_dims: the folded dimensions of the original tensor, which can be used to
             reverse the operation

    Examples:
        >>> x = torch.ones(10, 20, 5, 4, 8)
        >>> x, batch_dim = fold_batch(x, 2)
        >>> x.shape == (1000, 4, 8)
        >>> batch_dim == (10, 20, 5)

    """
    if nonbatch_ndims > 0:
        batch_dims = x.shape[:-nonbatch_ndims]
        x = x.view(-1, *(x.shape[-nonbatch_ndims:]))
        return x, batch_dims
    else:
        batch_dims = x.shape
        x = x.view(-1)
        return x, batch_dims


def unfold_batch(x: Tensor, batch_dims: Union[Size, Tuple]) -> Tensor:
    r"""
    Overview:
        Unfold the batch dimension of a tensor.

    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to unfold
        - batch_dims (:obj:`torch.Size`): the dimensions that are folded

    Returns:
        - x (:obj:`torch.Tensor`): the original unfolded tensor

    Examples:
        >>> x = torch.ones(10, 20, 5, 4, 8)
        >>> x, batch_dim = fold_batch(x, 2)
        >>> x.shape == (1000, 4, 8)
        >>> batch_dim == (10, 20, 5)
        >>> x = unfold_batch(x, batch_dim)
        >>> x.shape == (10, 20, 5, 4, 8)
    """
    return x.view(*batch_dims, *x.shape[1:])


def unsqueeze_repeat(x: Tensor, repeat_times: int, unsqueeze_dim: int = 0) -> Tensor:
    r"""
    Overview:
        Squeeze the tensor on `unsqueeze_dim` and then repeat in this dimension for `repeat_times` times.\
        This is useful for preproprocessing the input to an model ensemble.

    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to squeeze and repeat
        - repeat_times (:obj:`int`): the times that the tensor is repeatd
        - unsqueeze_dim (:obj:`int`): the unsqueezed dimension

    Returns:
        - x (:obj:`torch.Tensor`): the unsqueezed and repeated tensor

    Examples:
        >>> x = torch.ones(64, 6)
        >>> x = unsqueeze_repeat(x, 4)
        >>> x.shape == (4, 64, 6)

        >>> x = torch.ones(64, 6)
        >>> x = unsqueeze_repeat(x, 4, -1)
        >>> x.shape == (64, 6, 4)
    """
    assert -1 <= unsqueeze_dim <= len(x.shape), f'unsqueeze_dim should be from {-1} to {len(x.shape)}'
    x = x.unsqueeze(unsqueeze_dim)
    repeats = [1] * len(x.shape)
    repeats[unsqueeze_dim] *= repeat_times
    return x.repeat(*repeats)

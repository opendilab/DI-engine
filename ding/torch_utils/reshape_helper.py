from typing import Tuple, Union

from torch import Tensor, Size


def fold_batch(x: Tensor, nonbatch_dims=1) -> Tuple[Tensor, Size]:
    """
    Overview:
        Fold the first (dim - nonbatch_dims) dimensions of a tensor as batch dimension.\
            This operation is similar to torch.flatten but provides an inverse function\
            `unfold_batch` to restore the folded dimensions.
    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to fold
        - nonbatch_dims (:obj:`int`): the number of dimensions that is not folded as\
            batch dimension.
    Returns:
        - x (:obj:`torch.Tensor`): the folded tensor
        - batch_dim: the folded dimension of the original tensor, which can be used to\
             reverse the operation
    Examples:
        # (T, B, X) => (T*B, X)
        >>> x = torch.ones(10, 20, 4, 8)
        >>> x, batch_dim = fold_batch(x, 2)
        >>> x.shape == (200, 4, 8)
        >>> batch_dim == (10, 20)
    """
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = x.view(-1, *(x.shape[-nonbatch_dims:]))
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = x.view(-1)
        return x, batch_dim


def unfold_batch(x: Tensor, batch_dim: Union[Size, Tuple]) -> Tensor:
    """
    Overview:
        Unfold the batch dimension of a tensor.
        (T*B, X) => (T, B, X)
    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to unfold
        - batch_dim (:obj:`torch.Size`): the dimensions that are folded
    Returns:
        - x (:obj:`torch.Tensor`): the original unfolded tensor
    Examples:
        >>> x = torch.ones(10, 20, 4, 8)
        >>> x, batch_dim = fold_batch(x, 2)
        >>> x.shape == (200, 4, 8)
        >>> batch_dim == (10, 20)
        >>> x = unfold_batch(x, batch_dim)
        >>> x.shape == (10, 20, 4, 8)
    """
    return x.view(*batch_dim, *x.shape[1:])

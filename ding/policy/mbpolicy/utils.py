from typing import Callable, Tuple, Union

from torch import Tensor, Size


def flatten_batch(x: Tensor, nonbatch_dims=1) -> Tuple[Tensor, Size]:
    """
    Overview:
        Flatten the first (dim - nonbatch_dims) dimensions of a tensor as batch dimension.
        (T, B, X) => (T*B, X)
    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to flatten
        - nonbatch_dims (:obj:`int`): the number of dimensions that is not flattened as\
            batch dimension.z
    Returns:
        - x (:obj:`torch.Tensor`): the flattened tensor
        - batch_dim: the flattened dimension of the original tensor, which can be used to\
             reverse the operation.

    Examples::

        >>> x = torch.ones(10, 20, 4, 8)
        >>> x, batch_dim = flatten_batch(x, 2)
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


def unflatten_batch(x: Tensor, batch_dim: Union[Size, Tuple]) -> Tensor:
    """
    Overview:
        Unflatten the batch dimension of a tensor.
        (T*B, X) => (T, B, X)
    Arguments:
        - x (:obj:`torch.Tensor`): the tensor to unflatten
        - batch_dim (:obj:`torch.Size`): the dimensions that are flattened
    Returns:
        - x (:obj:`torch.Tensor`): the original unflattened tensor

    Examples::

        >>> x = torch.ones(10, 20, 4, 8)
        >>> x, batch_dim = flatten_batch(x, 2)
        >>> x.shape == (200, 4, 8)
        >>> batch_dim == (10, 20)
        >>> x = unflatten_batch(x, batch_dim)
        >>> x.shape == (10, 20, 4, 8)
    """
    return x.view(*batch_dim, *x.shape[1:])


def q_evaluation(obss: Tensor, actions: Tensor, q_critic_fn: Callable[[Tensor, Tensor],
                                                                      Tensor]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Overview:
        Evaluate observation&action pairs along the trajectory
    Arguments:
        - obss (:obj:`torch.Tensor`): the observations along the trajectory
        - actions (:obj:`torch.Size`): the actions along the trajectory
        - q_critic_fn (:obj:`Callable`): the unified API Q(S_t, A_t)
    Returns:
        - q_value (:obj:`torch.Tensor`): the action-value function evaluated along the trajectory
    Shapes:
        N: time step
        B: batch size
        O: observation dimension
        A: action dimension

        - obss:        [N, B, O]
        - actions:     [N, B, A]
        - q_value:     [N, B]

    """
    obss, dim = flatten_batch(obss, 1)
    actions, _ = flatten_batch(actions, 1)
    q_values = q_critic_fn(obss, actions)
    # twin critic
    if isinstance(q_values, list):
        return [unflatten_batch(q_values[0], dim), unflatten_batch(q_values[1], dim)]
    return unflatten_batch(q_values, dim)

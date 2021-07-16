import torch
from typing import Optional, Callable


def levenshtein_distance(
        pred: torch.LongTensor,
        target: torch.LongTensor,
        pred_extra: Optional[torch.Tensor] = None,
        target_extra: Optional[torch.Tensor] = None,
        extra_fn: Optional[Callable] = None
) -> torch.FloatTensor:
    r"""
    Overview:
        Levenshtein Distance, i.e. Edit Distance.
    Arguments:
        - pred (:obj:`torch.LongTensor`): shape: (N1, )  (N1 >= 0)
        - target (:obj:`torch.LongTensor`): shape: (N2, )  (N2 >= 0)
        - pred_extra (:obj:`Optional[torch.Tensor]`)
        - target_extra (:obj:`Optional[torch.Tensor]`)
        - extra_fn (:obj:`Optional[Callable]`): if specified, the distance metric of the extra input data
    Returns:
        - distance (:obj:`torch.FloatTensor`): distance(scalar), shape: (1, )
    """
    assert (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor))
    assert (pred.dtype == torch.long and target.dtype == torch.long), '{}\t{}'.format(pred.dtype, target.dtype)
    assert (pred.device == target.device)
    assert (type(pred_extra) == type(target_extra))
    if not extra_fn:
        assert (not pred_extra)
    N1, N2 = pred.shape[0], target.shape[0]
    assert (N1 >= 0 and N2 >= 0)
    if N1 == 0 or N2 == 0:
        distance = max(N1, N2)
    else:
        dp_array = torch.zeros(N1, N2).float()
        if extra_fn:
            if pred[0] == target[0]:
                extra = extra_fn(pred_extra[0], target_extra[0])
            else:
                extra = 1.
            dp_array[0, :] = torch.arange(0, N2) + extra
            dp_array[:, 0] = torch.arange(0, N1) + extra
        else:
            dp_array[0, :] = torch.arange(0, N2)
            dp_array[:, 0] = torch.arange(0, N1)
        for i in range(1, N1):
            for j in range(1, N2):
                if pred[i] == target[j]:
                    if extra_fn:
                        dp_array[i, j] = dp_array[i - 1, j - 1] + extra_fn(pred_extra[i], target_extra[j])
                    else:
                        dp_array[i, j] = dp_array[i - 1, j - 1]
                else:
                    dp_array[i, j] = min(dp_array[i - 1, j] + 1, dp_array[i, j - 1] + 1, dp_array[i - 1, j - 1] + 1)
        distance = dp_array[N1 - 1, N2 - 1]
    return torch.FloatTensor([distance]).to(pred.device)


def hamming_distance(pred: torch.LongTensor, target: torch.LongTensor, weight=1.) -> torch.LongTensor:
    r'''
    Overview:
        Hamming Distance
    Arguments:
        - pred (:obj:`torch.LongTensor`): pred input, boolean vector(0 or 1)
        - target (:obj:`torch.LongTensor`): target input, boolean vector(0 or 1)
        - weight (:obj:`torch.LongTensor`): weight to multiply
    Returns:
        - distance(:obj:`torch.LongTensor`): distance(scalar), shape (1, )
    Shapes:
        - pred & target (:obj:`torch.LongTensor`): shape :math:`(B, N)`, \
            while B is the batch size, N is the dimension
    '''
    assert (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor))
    assert (pred.dtype == torch.long and target.dtype == torch.long)
    assert (pred.device == target.device)
    assert (pred.shape == target.shape)
    return pred.ne(target).sum(dim=1).float().mul_(weight)

from typing import Optional
import torch


def cov(
        x: torch.Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: Optional[int] = None,
        aweights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Overview:
        Estimates covariance matrix like ``numpy.cov``
    Arguments:
        - x (:obj:`torch.Tensor`)
        - rowvar (:obj:`bool`)
        - bias (:obj:`bool`)
        - ddof (:obj:`Optional[int]`)
        - aweights (:obj:`Optional[torch.Tensor]`)
    Returns:
        - cov_mat (:obj:`torch.Tensor`): Covariance matrix
    """
    if x.dim() == 1 and rowvar:
        raise NotImplementedError
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    # elif aweights is None:
    #     fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

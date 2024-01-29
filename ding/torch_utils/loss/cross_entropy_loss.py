import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional


class LabelSmoothCELoss(nn.Module):
    """
    Overview:
        Label smooth cross entropy loss.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, ratio: float) -> None:
        """
        Overview:
            Initialize the LabelSmoothCELoss object using the given arguments.
        Arguments:
            - ratio (:obj:`float`): The ratio of label-smoothing (the value is in 0-1). If the ratio is larger, the \
                extent of label smoothing is larger.
        """
        super().__init__()
        self.ratio = ratio

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """
        Overview:
            Calculate label smooth cross entropy loss.
        Arguments:
            - logits (:obj:`torch.Tensor`): Predicted logits.
            - labels (:obj:`torch.LongTensor`): Ground truth.
        Returns:
            - loss (:obj:`torch.Tensor`): Calculated loss.
        """
        B, N = logits.shape
        val = float(self.ratio) / (N - 1)
        one_hot = torch.full_like(logits, val)
        one_hot.scatter_(1, labels.unsqueeze(1), 1 - val)
        logits = F.log_softmax(logits, dim=1)
        return -torch.sum(logits * (one_hot.detach())) / B


class SoftFocalLoss(nn.Module):
    """
    Overview:
        Soft focal loss.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self, gamma: int = 2, weight: Any = None, size_average: bool = True, reduce: Optional[bool] = None
    ) -> None:
        """
        Overview:
            Initialize the SoftFocalLoss object using the given arguments.
        Arguments:
            - gamma (:obj:`int`): The extent of focus on hard samples. A smaller ``gamma`` will lead to more focus on \
                easy samples, while a larger ``gamma`` will lead to more focus on hard samples.
            - weight (:obj:`Any`): The weight for loss of each class.
            - size_average (:obj:`bool`): By default, the losses are averaged over each loss element in the batch. \
                Note that for some losses, there are multiple elements per sample. If the field ``size_average`` is \
                set to ``False``, the losses are instead summed for each minibatch. Ignored when ``reduce`` is \
                ``False``.
            - reduce (:obj:`Optional[bool]`): By default, the losses are averaged or summed over observations for \
                each minibatch depending on size_average. When ``reduce`` is ``False``, returns a loss for each batch \
                element instead and ignores ``size_average``.
        """
        super().__init__()
        self.gamma = gamma
        self.nll_loss = torch.nn.NLLLoss2d(weight, size_average, reduce=reduce)

    def forward(self, inputs: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """
        Overview:
            Calculate soft focal loss.
        Arguments:
            - logits (:obj:`torch.Tensor`): Predicted logits.
            - labels (:obj:`torch.LongTensor`): Ground truth.
        Returns:
            - loss (:obj:`torch.Tensor`): Calculated loss.
        """
        return self.nll_loss((1 - F.softmax(inputs, 1)) ** self.gamma * F.log_softmax(inputs, 1), targets)


def build_ce_criterion(cfg: dict) -> nn.Module:
    """
    Overview:
        Get a cross entropy loss instance according to given config.
    Arguments:
        - cfg (:obj:`dict`) : Config dict. It contains:
            - type (:obj:`str`): Type of loss function, now supports ['cross_entropy', 'label_smooth_ce', \
                'soft_focal_loss'].
            - kwargs (:obj:`dict`): Arguments for the corresponding loss function.
    Returns:
        - loss (:obj:`nn.Module`): loss function instance
    """
    if cfg.type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif cfg.type == 'label_smooth_ce':
        return LabelSmoothCELoss(cfg.kwargs.smooth_ratio)
    elif cfg.type == 'soft_focal_loss':
        return SoftFocalLoss()
    else:
        raise ValueError("invalid criterion type:{}".format(cfg.type))

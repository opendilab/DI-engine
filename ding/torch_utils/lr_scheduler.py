from functools import partial
import math

import torch.optim
from torch.optim.lr_scheduler import LambdaLR


def get_lr_ratio(epoch: int, warmup_epochs: int, learning_rate: float, lr_decay_epochs: int, min_lr: float) -> float:
    """
    Overview:
        Get learning rate ratio for each epoch.
    Arguments:
        - epoch (:obj:`int`): Current epoch.
        - warmup_epochs (:obj:`int`): Warmup epochs.
        - learning_rate (:obj:`float`): Learning rate.
        - lr_decay_epochs (:obj:`int`): Learning rate decay epochs.
        - min_lr (:obj:`float`): Minimum learning rate.
    """

    # 1) linear warmup for warmup_epochs.
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    # 2) if epoch> lr_decay_epochs, return min learning rate
    if epoch > lr_decay_epochs:
        return min_lr / learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (epoch - warmup_epochs) / (lr_decay_epochs - warmup_epochs)
    assert 0 <= decay_ratio <= 1
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr + coefficient * (learning_rate - min_lr)) / learning_rate


def cos_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        warmup_epochs: float = 5,
        lr_decay_epochs: float = 100,
        min_lr: float = 6e-5
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Overview:
        Cosine learning rate scheduler.
    Arguments:
        - optimizer (:obj:`torch.optim.Optimizer`): Optimizer.
        - learning_rate (:obj:`float`): Learning rate.
        - warmup_epochs (:obj:`float`): Warmup epochs.
        - lr_decay_epochs (:obj:`float`): Learning rate decay epochs.
        - min_lr (:obj:`float`): Minimum learning rate.
    """

    return LambdaLR(
        optimizer,
        partial(
            get_lr_ratio,
            warmup_epochs=warmup_epochs,
            lr_decay_epochs=lr_decay_epochs,
            min_lr=min_lr,
            learning_rate=learning_rate
        )
    )

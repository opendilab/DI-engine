from functools import partial
import math

from torch.optim.lr_scheduler import LambdaLR


def get_lr(it, warmup_epochs, learning_rate, lr_decay_epochs, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_epochs:
        return it / warmup_epochs
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_epochs:
        return min_lr / learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_epochs) / (lr_decay_epochs - warmup_epochs)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return (min_lr + coeff * (learning_rate - min_lr)) / learning_rate


def cos_lr_scheduler(optimizer, learning_rate, warmup_epochs=5, lr_decay_epochs=100, min_lr=6e-5):
    return LambdaLR(optimizer, partial(get_lr, warmup_epochs=warmup_epochs, lr_decay_epochs=lr_decay_epochs,
                                       min_lr=min_lr, learning_rate=learning_rate))

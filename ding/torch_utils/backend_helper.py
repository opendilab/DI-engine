import torch


def enable_tf32() -> None:
    """
    Overview:
        Enable tf32 on matmul and cudnn for faster computation. This only works on Ampere GPU devices. \
        For detailed information, please refer to: \
        https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices.
    """
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

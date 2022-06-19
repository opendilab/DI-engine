import torch


def torch_ge_131():
    return int("".join(list(filter(str.isdigit, torch.__version__)))) >= 131


def torch_ge_180():
    return int("".join(list(filter(str.isdigit, torch.__version__)))) >= 180

import torch


def torch_gt_131():
    return int("".join(list(filter(str.isdigit, torch.__version__)))) >= 131

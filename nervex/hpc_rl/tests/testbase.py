import torch
import numpy as np

torch.set_printoptions(precision=6)

times = 6


def mean_relative_error(y_true, y_pred):
    eps = 1e-5
    relative_error = np.average(np.abs(y_true - y_pred) / (y_true + eps), axis=0)
    return relative_error

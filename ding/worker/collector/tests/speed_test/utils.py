import numpy as np


def random_change(number):
    return number * (1 + (np.random.random() - 0.5) * 0.6)

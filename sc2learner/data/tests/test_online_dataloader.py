import copy
from collections import defaultdict

import numpy as np
import pytest
import torch

from sc2learner.data.online import OnlineIteratorDataLoader


def make_dataset():
    index = 0
    while True:
        yield [index]
        index += 1


def test_online():
    def read_data_fn(data):
        return torch.tensor([[1,2,3],[4,5,6]])

    dataloader = OnlineIteratorDataLoader(
        iter(make_dataset()), batch_size=1, read_data_fn=read_data_fn, num_workers=3
    )

    i = 0
    while i < 100:
        batch_data = next(dataloader)
        print(i, batch_data.tolist())
        i += 1


if __name__ == '__main__':
    test_online()


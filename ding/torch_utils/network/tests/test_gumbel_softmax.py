import numpy as np
import pytest
import torch

from ding.torch_utils.network import GumbelSoftmax, gumbel_softmax


@pytest.mark.unittest
class TestGumbelSoftmax:

    def test(self):
        B = 4
        N = 10
        model = GumbelSoftmax()
        # data case 1
        for _ in range(N):
            data = torch.rand((4, 10))
            data = torch.log(data)
            gumbelsoftmax = model(data, hard=False)
            assert gumbelsoftmax.shape == (B, N)
        # data case 2
        for _ in range(N):
            data = torch.rand((4, 10))
            data = torch.log(data)
            gumbelsoftmax = model(data, hard=True)
            assert gumbelsoftmax.shape == (B, N)

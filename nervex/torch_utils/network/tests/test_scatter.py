import pytest
import numpy as np
import torch
from nervex.torch_utils import ScatterConnection


@pytest.mark.unittest
class TestScatterConnection:

    def test_naive(self):
        model = ScatterConnection()
        B, M, N = 4, 22, 256
        while True:
            try:
                inputs = torch.randn(B, M, N).requires_grad_(True)
                assert len(torch.nonzero(inputs)) == B * M * N
                break
            except AssertionError:
                continue

        H, W = 72, 96
        location = []
        for _ in range(B):
            tmp = [np.random.choice(range(H), M, replace=False), np.random.choice(range(W), M, replace=False)]
            location.append(list(zip(*tmp)))
        location = torch.LongTensor(location)
        output = model(inputs, (H, W), location)
        idx = torch.nonzero(output)
        assert len(idx) == B * M * N
        assert output.shape == (B, N, H, W)
        loss = output.mean()
        assert inputs.grad is None
        loss.backward()
        assert isinstance(inputs.grad, torch.Tensor)

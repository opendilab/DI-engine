import pytest
import torch
from ding.torch_utils import ScatterConnection


@pytest.mark.unittest
class TestScatterConnection:

    def test_naive(self):
        for scatter_type in ['add', 'cover']:
            model = ScatterConnection(scatter_type)
            B, M, N = 2, 24, 32
            H, W = 2, 3
            input = torch.rand(B, M, N).requires_grad_(True)
            h = torch.randint(
                low=0, high=H, size=(
                    B,
                    M,
                )
            ).unsqueeze(dim=2)
            w = torch.randint(
                low=0, high=W, size=(
                    B,
                    M,
                )
            ).unsqueeze(dim=2)
            location = torch.cat([h, w], dim=2)
            output = model(x=input, spatial_size=(H, W), location=location)
            loss = output.mean()
            loss.backward()
            assert isinstance(input.grad, torch.Tensor)

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

    def test_xy_forward(self):
        for scatter_type in ['add', 'cover']:
            model = ScatterConnection(scatter_type)
            BatchSize, Num, EmbeddingSize = 10, 20, 3
            SpatialSize = (13, 17)
            input = torch.randn(size=(BatchSize, Num, EmbeddingSize)).requires_grad_(True)
            coord_x = torch.randint(low=0, high=13, size=(BatchSize, Num))
            coord_y = torch.randint(low=0, high=17, size=(BatchSize, Num))
            output = model.xy_forward(input, SpatialSize, coord_x, coord_y)
            loss = output.mean()
            loss.backward()
            assert isinstance(input.grad, torch.Tensor)
            assert output.shape == (BatchSize, EmbeddingSize, *SpatialSize)

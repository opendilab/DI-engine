import torch
import pytest
from ding.model import ScatterConnection

BatchSize, Num, EmbeddingSize = 10, 20, 3
SpatialSize = (13, 17)


@pytest.mark.unittest
class TestScatterConnection:

    def test_scatter_connection_forward(self):
        scatter_conn = ScatterConnection()
        for _ in range(10):
            x = torch.randn(size=(BatchSize, Num, EmbeddingSize))
            locations = torch.randint(low=0, high=13, size=(BatchSize, Num, 2))
            outputs = scatter_conn.forward(x, SpatialSize, location=locations)
            assert outputs.shape == (BatchSize, EmbeddingSize, *SpatialSize)

    def test_scatter_connection_xy_forward(self):
        scatter_conn = ScatterConnection()
        for _ in range(10):
            x = torch.randn(size=(BatchSize, Num, EmbeddingSize))
            coord_x = torch.randint(low=0, high=13, size=(BatchSize, Num))
            coord_y = torch.randint(low=0, high=17, size=(BatchSize, Num))
            outputs = scatter_conn.xy_forward(x, SpatialSize, coord_x, coord_y)
            assert outputs.shape == (BatchSize, EmbeddingSize, *SpatialSize)

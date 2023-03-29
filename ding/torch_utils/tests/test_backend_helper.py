import pytest
import torch

from ding.torch_utils.backend_helper import enable_tf32


@pytest.mark.unittest
class TestBackendHelper:

    def test_tf32(self):
        r"""
        Overview:
            Test the tf32.
        """
        enable_tf32()
        net = torch.nn.Linear(3, 4)
        x = torch.randn(1, 3)
        y = torch.sum(net(x))
        y.backward()

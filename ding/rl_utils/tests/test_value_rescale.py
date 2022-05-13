import pytest
import torch
from ding.rl_utils.value_rescale import value_inv_transform, value_transform


@pytest.mark.unittest
class TestValueRescale:

    def test_value_transform(self):
        for _ in range(10):
            t = torch.rand((2, 3))
            assert isinstance(value_transform(t), torch.Tensor)
            assert value_transform(t).shape == t.shape

    def test_value_inv_transform(self):
        for _ in range(10):
            t = torch.rand((2, 3))
            assert isinstance(value_inv_transform(t), torch.Tensor)
            assert value_inv_transform(t).shape == t.shape

    def test_trans_inverse(self):
        for _ in range(10):
            t = torch.rand((4, 16))
            diff = value_inv_transform(value_transform(t)) - t
            assert pytest.approx(diff.abs().max().item(), abs=2e-5) == 0
            assert pytest.approx(diff.abs().max().item(), abs=2e-5) == 0

import pytest
import torch
from ding.rl_utils.value_rescale import value_inv_transform, value_transform, symlog, inv_symlog


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


@pytest.mark.unittest
class TestSymlog:

    def test_symlog(self):
        for _ in range(10):
            t = torch.rand((3, 4))
            assert isinstance(symlog(t), torch.Tensor)
            assert symlog(t).shape == t.shape

    def test_inv_symlog(self):
        for _ in range(10):
            t = torch.rand((3, 4))
            assert isinstance(inv_symlog(t), torch.Tensor)
            assert inv_symlog(t).shape == t.shape

    def test_trans_inverse(self):
        for _ in range(10):
            t = torch.rand((4, 16))
            diff = inv_symlog(symlog(t)) - t
            assert pytest.approx(diff.abs().max().item(), abs=2e-5) == 0
            assert pytest.approx(diff.abs().max().item(), abs=2e-5) == 0

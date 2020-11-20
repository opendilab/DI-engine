import pytest
import torch
from nervex.rl_utils.value_rescale import value_inverse_h, value_transition_h


@pytest.mark.unittest
class TestValueRescale:

    def test_value_transition_h(self):
        for _ in range(10):
            t = torch.rand((2, 3))
            assert isinstance(value_transition_h(t), torch.Tensor)
            assert value_transition_h(t).shape == t.shape

    def test_value_inverse_h(self):
        for _ in range(10):
            t = torch.rand((2, 3))
            assert isinstance(value_inverse_h(t), torch.Tensor)
            assert value_inverse_h(t).shape == t.shape

    def test_trans_inverse(self):
        for _ in range(10):
            t = torch.rand((2, 3))
            diff = value_inverse_h(value_transition_h(t)) - t
            assert diff.abs().mean().item() < 1e-5

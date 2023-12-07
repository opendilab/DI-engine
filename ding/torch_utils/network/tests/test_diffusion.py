import pytest
import torch
from ding.torch_utils.network.diffusion import DiffusionUNet1d, TemporalValue

batch_size = 2
transition_dim = 10
dim = 8
dim_mults = [1, 2, 4]
horizon = 4


@pytest.mark.unittest
class TestDiffusionNet:

    def test_DiffusionNet1d(self):
        diffusion = DiffusionUNet1d(transition_dim, dim, dim_mults)
        input = torch.rand(batch_size, horizon, transition_dim)
        t = torch.randint(0, horizon, (batch_size, )).long()
        cond = {t: torch.randn(batch_size, 2) for t in range(horizon)}
        output = diffusion(input, cond, time=t)
        assert output.shape == (batch_size, horizon, transition_dim)

    def test_TemporalValue(self):
        value = TemporalValue(horizon, transition_dim, dim, dim_mults=dim_mults)
        input = torch.rand(batch_size, horizon, transition_dim)
        t = torch.randint(0, horizon, (batch_size, )).long()
        cond = {t: torch.randn(batch_size, 2) for t in range(horizon)}
        output = value(input, cond, time=t)
        assert output.shape == (batch_size, 1)

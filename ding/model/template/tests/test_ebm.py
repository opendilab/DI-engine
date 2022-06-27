import pytest

import torch
import numpy as np
from ding.model.template.ebm import EBM, AutoregressiveEBM, DFO, AutoRegressiveDFO

# batch, negative_samples, obs_shape, action_shape
B, N, O, A = 32, 1024, 11, 3

@pytest.mark.unittest
class TestEBM:

    def test_forward(self):
        obs = torch.randn(B, O)
        action = torch.randn(B, N, A)
        ebm = EBM(O, A)
        energy = ebm(obs, action)
        assert energy.shape == (B, N)

@pytest.mark.unittest
class TestDFO:
    opt = DFO()
    opt.set_action_bounds(np.stack([np.zeros(A), np.ones(A)], axis=0))
    ebm = EBM(O, A)

    def test_sample(self):
        assert self.opt.sample(B, self.ebm).shape == (B, 256, A)

    def test_infer(self):
        obs = torch.randn(B, O)
        action = self.opt.infer(obs, self.ebm)
        assert action.shape == (B, A)

@pytest.mark.unittest
class TestAutoregressiveEBM:

    def test_forward(self):
        obs = torch.randn(B, O)
        action = torch.randn(B, N, A)
        arebm = AutoregressiveEBM(O, A)
        energy = arebm(obs, action)
        assert energy.shape == (B, N, A)

@pytest.mark.unittest
class TestAutoregressiveDFO:
    opt = AutoRegressiveDFO()
    opt.set_action_bounds(np.stack([np.zeros(A), np.ones(A)], axis=0))
    ebm = AutoregressiveEBM(O, A)

    def test_sample(self):
        assert self.opt.sample(B, self.ebm).shape == (B, 256, A)

    def test_infer(self):
        obs = torch.randn(B, O)
        action = self.opt.infer(obs, self.ebm)
        assert action.shape == (B, A)
import pytest

import torch
import numpy as np
from ding.model.template.ebm import EBM, AutoregressiveEBM
from ding.model.template.ebm import DFO, AutoRegressiveDFO, MCMC

# batch, negative_samples, obs_shape, action_shape
B, N, O, A = 32, 1024, 11, 3


@pytest.mark.unittest
class TestEBM:

    def test_forward(self):
        obs = torch.randn(B, N, O)
        action = torch.randn(B, N, A)
        ebm = EBM(O, A)
        energy = ebm(obs, action)
        assert energy.shape == (B, N)


@pytest.mark.unittest
class TestDFO:
    opt = DFO(train_samples=N, inference_samples=N)
    opt.set_action_bounds(np.stack([np.zeros(A), np.ones(A)], axis=0))
    ebm = EBM(O, A)

    def test_sample(self):
        obs = torch.randn(B, O)
        tiled_obs, action_samples = self.opt.sample(obs, self.ebm)
        assert tiled_obs.shape == (B, N, O)
        assert action_samples.shape == (B, N, A)

    def test_infer(self):
        obs = torch.randn(B, O)
        action = self.opt.infer(obs, self.ebm)
        assert action.shape == (B, A)


@pytest.mark.unittest
class TestAutoregressiveEBM:

    def test_forward(self):
        obs = torch.randn(B, N, O)
        action = torch.randn(B, N, A)
        arebm = AutoregressiveEBM(O, A)
        energy = arebm(obs, action)
        assert energy.shape == (B, N, A)


@pytest.mark.unittest
class TestAutoregressiveDFO:
    opt = AutoRegressiveDFO(train_samples=N, inference_samples=N)
    opt.set_action_bounds(np.stack([np.zeros(A), np.ones(A)], axis=0))
    ebm = AutoregressiveEBM(O, A)

    def test_sample(self):
        obs = torch.randn(B, O)
        tiled_obs, action_samples = self.opt.sample(obs, self.ebm)
        assert tiled_obs.shape == (B, N, O)
        assert action_samples.shape == (B, N, A)

    def test_infer(self):
        obs = torch.randn(B, O)
        action = self.opt.infer(obs, self.ebm)
        assert action.shape == (B, A)


@pytest.mark.unittest
class TestMCMC:
    opt = MCMC(iters=3, train_samples=N, inference_samples=N)
    opt.set_action_bounds(np.stack([np.zeros(A), np.ones(A)], axis=0))
    obs = torch.randn(B, N, O)
    action = torch.randn(B, N, A)
    ebm = EBM(O, A)

    def test_gradient_wrt_act(self):
        ebm = EBM(O, A)
        # inference mode
        de_dact = MCMC._gradient_wrt_act(self.obs, self.action, ebm)
        assert de_dact.shape == (B, N, A)
        # train mode
        de_dact = MCMC._gradient_wrt_act(self.obs, self.action, ebm, create_graph=True)
        loss = de_dact.pow(2).sum()
        loss.backward()
        assert de_dact.shape == (B, N, A)
        assert ebm.net[0].weight.grad is not None

    def test_langevin_step(self):
        stepsize = 1
        action = self.opt._langevin_step(self.obs, self.action, stepsize, self.ebm)
        assert action.shape == (B, N, A)
        # TODO: new action should have lower energy

    def test_langevin_action_given_obs(self):
        action = self.opt._langevin_action_given_obs(self.obs, self.action, self.ebm)
        assert action.shape == (B, N, A)

    def test_grad_penalty(self):
        ebm = EBM(O, A)
        self.opt.add_grad_penalty = True
        loss = self.opt.grad_penalty(self.obs, self.action, ebm)
        loss.backward()
        assert ebm.net[0].weight.grad is not None

    def test_sample(self):
        obs = torch.randn(B, O)
        tiled_obs, action_samples = self.opt.sample(obs, self.ebm)
        assert tiled_obs.shape == (B, N, O)
        assert action_samples.shape == (B, N, A)

    def test_infer(self):
        obs = torch.randn(B, O)
        action = self.opt.infer(obs, self.ebm)
        assert action.shape == (B, A)

import torch
import numpy as np
import pytest

from nervex.model import FCDiscreteNet, ConvDiscreteNet, FCRDiscreteNet, ConvRDiscreteNet, NoiseFCDiscreteNet, \
    NoiseDistributionFCDiscreteNet, NoiseQuantileFCDiscreteNet
from nervex.torch_utils import is_differentiable

B = 4
T = 6
embedding_dim = 32
action_dim_args = [(6, ), [4, 8], [
    1,
]]


@pytest.mark.unittest
@pytest.mark.parametrize('action_dim', action_dim_args)
class TestDiscreteNet:

    def output_check(self, model, outputs):
        action_dim = model._head.action_dim
        if isinstance(action_dim, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_dim):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_fc_discrete_net(self, action_dim):
        N = 32
        inputs = {'obs': torch.randn(B, N)}
        model = FCDiscreteNet((N, ), action_dim, [128, embedding_dim])
        outputs = model(inputs)['logit']
        self.output_check(model, outputs)

    def test_noise_fc_discrete_net(self, action_dim):
        N = 32
        inputs = {'obs': torch.randn(B, N)}
        model = NoiseFCDiscreteNet((N, ), action_dim, [128, embedding_dim])
        outputs = model(inputs)['logit']
        self.output_check(model, outputs)

    def test_noise_distribution_fc_discrete_net(self, action_dim):
        N = 32
        inputs = {'obs': torch.randn(B, N)}
        model = NoiseDistributionFCDiscreteNet((N, ), action_dim, [128, embedding_dim])
        outputs = model(inputs)['logit']
        self.output_check(model, outputs)

    def test_noise_quantile_fc_discrete_net(self, action_dim):
        N = 32
        inputs = {'obs': torch.randn(B, N)}
        model = NoiseQuantileFCDiscreteNet((N, ), action_dim, [128, embedding_dim])
        outputs = model(inputs)['logit']
        self.output_check(model, outputs)

    def test_conv_discrete_net(self, action_dim):
        dims = [3, 64, 64]
        inputs = torch.randn(B, *dims)
        model = ConvDiscreteNet(dims, action_dim, [128, embedding_dim])
        outputs = model(inputs)['logit']
        self.output_check(model, outputs)

    def test_fc_r_discrete_net(self, action_dim):
        N = 32
        data = torch.randn(T, B, N)
        model = FCRDiscreteNet((N, ), action_dim, [128, embedding_dim])
        prev_state = [None for _ in range(B)]
        for t in range(T):
            inputs = {'obs': data[t], 'prev_state': prev_state}
            outputs = model(inputs)
            logit, prev_state = outputs['logit'], outputs['next_state']
            assert len(prev_state) == B
            assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
        # test the last step can backward correctly
        self.output_check(model, logit)

        model = FCRDiscreteNet((N, ), action_dim, [128, embedding_dim])
        data = torch.randn(T, B, N)
        prev_state = [None for _ in range(B)]
        inputs = {'obs': data, 'prev_state': prev_state, 'enable_fast_timestep': True}
        outputs = model(inputs)
        logit, prev_state = outputs['logit'], outputs['next_state']
        assert len(prev_state) == B
        assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
        self.output_check(model, logit)
        action_dim = model._head.action_dim
        if isinstance(action_dim, tuple):
            assert all([l.shape == (T, B, d) for l, d in zip(logit, action_dim)])
        elif np.isscalar(action_dim):
            assert logit.shape == (T, B, action_dim)

    def test_conv_r_discrete_net(self, action_dim):
        dims = [3, 64, 64]
        data = torch.randn(T, B, *dims)
        model = ConvRDiscreteNet(dims, action_dim, [128, embedding_dim])
        prev_state = [None for _ in range(B)]
        for t in range(T):
            inputs = {'obs': data[t], 'prev_state': prev_state}
            outputs = model(inputs)
            logit, prev_state = outputs['logit'], outputs['next_state']
            assert len(prev_state) == B
            assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
        # test the last step can backward correctly
        self.output_check(model, logit)

        data = torch.randn(T, B, *dims)
        model = ConvRDiscreteNet(dims, action_dim, [128, embedding_dim])
        prev_state = [None for _ in range(B)]
        inputs = {'obs': data, 'prev_state': prev_state, 'enable_fast_timestep': True}
        outputs = model(inputs)
        logit, prev_state = outputs['logit'], outputs['next_state']
        assert len(prev_state) == B
        assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
        self.output_check(model, logit)
        action_dim = model._head.action_dim
        if isinstance(action_dim, tuple):
            assert all([l.shape == (T, B, d) for l, d in zip(logit, action_dim)])
        elif np.isscalar(action_dim):
            assert logit.shape == (T, B, action_dim)

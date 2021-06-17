import torch
import numpy as np
import pytest

from nervex.model.common.head import DuelingHead, ReparameterizationHead, MultiDiscreteHead
from nervex.torch_utils import is_differentiable

B = 4
T = 6
embedding_dim = 64
action_shape = 12


@pytest.mark.unittest
class TestHead:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    def test_dueling(self):
        inputs = torch.randn(B, embedding_dim)
        model = DuelingHead(embedding_dim, action_shape, 3, 3)
        outputs = model(inputs)['logit']
        self.output_check(model, outputs)
        assert outputs.shape == (B, action_shape)

    @pytest.mark.parametrize('action_shape', [1, 8])
    def test_reparameterization(self, action_shape):
        inputs = torch.randn(B, embedding_dim)
        for sigma_type in ['fixed', 'independent', 'conditioned']:
            if sigma_type == 'fixed':
                model = ReparameterizationHead(
                    embedding_dim, action_shape, sigma_type=sigma_type, fixed_sigma_value=0.5
                )
                outputs = model(inputs)
                mu, sigma = outputs['mu'], outputs['sigma']
                assert mu.shape == (B, action_shape) and sigma.shape == (B, action_shape)
                assert sigma.eq(torch.full((B, action_shape), 0.5)).all()
                self.output_check(model, outputs)
            elif sigma_type == 'independent':
                model = ReparameterizationHead(embedding_dim, action_shape, sigma_type=sigma_type)
                outputs = model(inputs)
                mu, sigma = outputs['mu'], outputs['sigma']
                assert mu.shape == (B, action_shape) and sigma.shape == (B, action_shape)
                self.output_check(model, outputs)
                assert model.log_sigma_param.grad is not None
            elif sigma_type == 'conditioned':
                model = ReparameterizationHead(embedding_dim, action_shape, sigma_type=sigma_type)
                outputs = model(inputs)
                mu, sigma = outputs['mu'], outputs['sigma']
                assert mu.shape == (B, action_shape) and sigma.shape == (B, action_shape)
                self.output_check(model, outputs)

    def test_multi_discrete_head(self):
        output_size_list = [2, 3, 7]
        head = MultiDiscreteHead(DuelingHead, embedding_dim, output_size_list, activation=torch.nn.Tanh())
        print(head)
        inputs = torch.randn(B, embedding_dim)
        outputs = head(inputs)
        assert isinstance(outputs, dict)
        self.output_check(head, outputs['logit'])
        for i, d in enumerate(output_size_list):
            assert outputs['logit'][i].shape == (B, d)

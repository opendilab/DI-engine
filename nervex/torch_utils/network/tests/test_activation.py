import pytest
import torch
from nervex.torch_utils import build_activation


@pytest.mark.unittest
class TestActivation:
    def test(self):
        act_type = 'relu'
        act = build_activation(act_type, inplace=True)
        act_type = 'prelu'
        act = build_activation(act_type)
        try:
            act = build_activation(act_type, inplace=True)
        except Exception as e:
            assert isinstance(e, AssertionError)
        act_type = 'glu'
        input_dim = 50
        output_dim = 150
        context_dim = 200
        act = build_activation(act_type)(input_dim=input_dim, output_dim=output_dim, context_dim=context_dim, input_type='fc')
        batch_size = 10
        input = torch.rand(batch_size, input_dim).requires_grad_(True)
        context = torch.rand(batch_size, context_dim).requires_grad_(True)
        output = act(input, context)
        assert output.shape == (batch_size, output_dim)
        loss = output.mean()
        loss.backward()
        assert isinstance(input.grad, torch.Tensor)

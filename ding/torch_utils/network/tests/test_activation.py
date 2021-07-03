import pytest
import torch
from ding.torch_utils import build_activation


@pytest.mark.unittest
class TestActivation:

    def test(self):
        act_type = 'relu'
        act = build_activation(act_type, inplace=True)
        act_type = 'prelu'
        act = build_activation(act_type)
        with pytest.raises(AssertionError):
            act = build_activation(act_type, inplace=True)
        with pytest.raises(KeyError):
            act = build_activation('xxxlu')
        act_type = 'glu'
        input_dim = 50
        output_dim = 150
        context_dim = 200
        act = build_activation(act_type
                               )(input_dim=input_dim, output_dim=output_dim, context_dim=context_dim, input_type='fc')
        batch_size = 10
        inputs = torch.rand(batch_size, input_dim).requires_grad_(True)
        context = torch.rand(batch_size, context_dim).requires_grad_(True)
        output = act(inputs, context)
        assert output.shape == (batch_size, output_dim)
        assert act.layer1.weight.grad is None
        loss = output.mean()
        loss.backward()
        assert isinstance(inputs.grad, torch.Tensor)
        assert isinstance(act.layer1.weight.grad, torch.Tensor)

        act = build_activation(act_type)(
            input_dim=input_dim, output_dim=output_dim, context_dim=context_dim, input_type='conv2d'
        )
        size = 16
        inputs = torch.rand(batch_size, input_dim, size, size)
        context = torch.rand(batch_size, context_dim, size, size)
        output = act(inputs, context)
        assert output.shape == (batch_size, output_dim, size, size)
        assert act.layer1.weight.grad is None
        loss = output.mean()
        loss.backward()
        assert isinstance(act.layer1.weight.grad, torch.Tensor)

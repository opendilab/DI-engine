import pytest
import torch
from ding.torch_utils import PopArt

batch_size = 4
input_features = 16
output_features = 4


@pytest.mark.unittest
class TestPopArt:

    def test_popart(self):
        input = torch.rand((batch_size, input_features)).requires_grad_(True)
        model = PopArt(input_features, output_features)
        output = model(input)
        loss = output['pred'].mean()
        loss.backward()
        assert isinstance(input.grad, torch.Tensor)

        # validate the shape of parameters and outputs
        assert output['pred'].shape == (batch_size, output_features)
        assert output['unnormalized_pred'].shape == (batch_size, output_features)
        assert model.mu.shape == torch.Size([output_features])
        assert model.sigma.shape == torch.Size([output_features])
        assert model.v.shape == torch.Size([output_features])

        # validate the normalization
        assert torch.all(torch.abs(output['pred']) <= 1)

        model.update_parameters(torch.rand(batch_size, output_features))

        # validate the non-empty of parameters
        assert not torch.all(torch.isnan(model.mu))
        assert not torch.all(torch.isnan(model.sigma))
        assert not torch.all(torch.isnan(model.v))

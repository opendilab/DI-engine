import pytest
import torch

from ding.torch_utils import Transformer


@pytest.mark.unittest
class TestTransformer:

    def test(self):
        batch_size = 2
        num_entries = 2
        C = 2
        masks = [None, torch.rand(batch_size, num_entries).round().bool()]
        for mask in masks:
            output_dim = 4
            model = Transformer(
                input_dim=C,
                head_dim=2,
                hidden_dim=3,
                output_dim=output_dim,
                head_num=2,
                mlp_num=2,
                layer_num=2,
            )
            input = torch.rand(batch_size, num_entries, C).requires_grad_(True)
            output = model(input, mask)
            loss = output.mean()
            loss.backward()
            assert isinstance(input.grad, torch.Tensor)
            assert output.shape == (batch_size, num_entries, output_dim)

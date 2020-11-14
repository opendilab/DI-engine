import numpy as np
import pytest
import torch

from nervex.torch_utils import Transformer


@pytest.mark.unittest
class TestTransformer:

    def test(self):
        batch_size = 20
        num_entries = 50
        C = 100
        mask = None
        output_dim = 256
        model = Transformer(input_dim=C,
                            head_dim=128,
                            hidden_dim=1024,
                            output_dim=output_dim,
                            head_num=2,
                            mlp_num=2,
                            layer_num=3,)
        inputs = torch.rand(batch_size, num_entries, C)
        outputs = model(inputs, mask)
        assert outputs.shape == (batch_size, num_entries, output_dim)

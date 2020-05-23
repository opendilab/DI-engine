import pytest
import numpy as np
import torch
from sc2learner.torch_utils import Transformer


@pytest.mark.unittest
class TestTransformer:
    def test(self):
        B = 4
        output_dim = 256
        inputs = []
        output_num_list = []
        for _ in range(B):
            N = np.random.randint(400, 600)
            inputs.append(torch.randn(N, 340))
            output_num_list.append(N)

        model = Transformer(340, output_dim=output_dim)
        outputs = model(inputs)
        assert isinstance(outputs, list)
        assert len(outputs) == B
        for o, n in zip(outputs, output_num_list):
            assert o.shape == (n, output_dim)

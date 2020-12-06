import torch
import numpy as np
import pytest

from nervex.model import ConvEncoder
from nervex.torch_utils import is_differentiable

B = 4
C, H, W = 3, 128, 128
embedding_dim = 64


@pytest.mark.unittest
class TestEncoder:

    def output_check(self, model, outputs):
        loss = outputs.sum()
        is_differentiable(loss, model)

    def test_conv_encoder(self):
        inputs = torch.randn(B, C, H, W)
        model = ConvEncoder((C, H, W), embedding_dim)
        outputs = model(inputs)
        self.output_check(model, outputs)
        assert outputs.shape == (B, embedding_dim)

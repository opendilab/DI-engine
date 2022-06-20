import torch
import numpy as np
import pytest

from ding.model import ConvEncoder, FCEncoder, IMPALAConvEncoder
from ding.torch_utils import is_differentiable

B = 4
C, H, W = 3, 128, 128


@pytest.mark.unittest
class TestEncoder:

    def output_check(self, model, outputs):
        loss = outputs.sum()
        is_differentiable(loss, model)

    def test_conv_encoder(self):
        inputs = torch.randn(B, C, H, W)
        model = ConvEncoder((C, H, W), hidden_size_list=[32, 48, 64, 64, 128], activation=torch.nn.Tanh())
        print(model)
        outputs = model(inputs)
        self.output_check(model, outputs)
        assert outputs.shape == (B, 128)

    def test_fc_encoder(self):
        inputs = torch.randn(B, 32)
        hidden_size_list = [128 for _ in range(3)]
        model = FCEncoder(32, hidden_size_list, res_block=True, activation=torch.nn.Tanh())
        print(model)
        outputs = model(inputs)
        self.output_check(model, outputs)
        assert outputs.shape == (B, hidden_size_list[-1])

        hidden_size_list = [64, 128, 256]
        model = FCEncoder(32, hidden_size_list, res_block=False, activation=torch.nn.Tanh())
        print(model)
        outputs = model(inputs)
        self.output_check(model, outputs)
        assert outputs.shape == (B, hidden_size_list[-1])

    def test_impalaconv_encoder(self):
        inputs = torch.randn(B, 3, 64, 64)
        model = IMPALAConvEncoder(obs_shape=(3, 64, 64))
        print(model)
        outputs = model(inputs)
        self.output_check(model, outputs)
        assert outputs.shape == (B, 256)

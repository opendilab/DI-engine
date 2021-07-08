import torch
import pytest
from ding.torch_utils.network import ResBlock, ResFCBlock

batch_size = 2
in_channels = 2
H, W = 2, 3
activation = torch.nn.ReLU()
norm_type = 'BN'
res_type = ['basic', 'bottleneck']


@pytest.mark.unittest
class TestResBlock:

    def test_res_blcok(self):
        input = torch.rand(batch_size, in_channels, 2, 3).requires_grad_(True)
        for r in res_type:
            model = ResBlock(in_channels, activation, norm_type, r)
            output = model(input)
            loss = output.mean()
            loss.backward()
            assert output.shape == input.shape
            assert isinstance(input.grad, torch.Tensor)

    def test_res_fc_block(self):
        input = torch.rand(batch_size, in_channels).requires_grad_(True)
        model = ResFCBlock(in_channels, activation, norm_type)
        output = model(input)
        loss = output.mean()
        loss.backward()
        assert output.shape == input.shape
        assert isinstance(input.grad, torch.Tensor)

import torch
import pytest
from ding.torch_utils import build_activation, build_normalization
from ding.torch_utils.network.nn_module import conv1d_block, conv2d_block, fc_block, deconv2d_block, ChannelShuffle, \
    one_hot, NearestUpsample, BilinearUpsample, binary_encode, weight_init_, NaiveFlatten, normed_linear, normed_conv2d

batch_size = 2
in_channels = 2
out_channels = 3
H = 2
W = 3
kernel_size = 2
stride = 1
padding = 0
dilation = 1
groups = 1
init_type = ['xavier', 'kaiming', 'orthogonal']
act = build_activation('relu')
norm_type = 'BN'


@pytest.mark.unittest
class TestNnModule:

    def run_model(self, input, model):
        output = model(input)
        loss = output.mean()
        loss.backward()
        assert isinstance(
            input.grad,
            torch.Tensor,
        )
        return output

    def test_weight_init(self):
        weight = torch.zeros(2, 3)
        for init_type in ['xavier', 'orthogonal']:
            weight_init_(weight, init_type)
        for act in [torch.nn.LeakyReLU(), torch.nn.ReLU()]:
            weight_init_(weight, 'kaiming', act)
        with pytest.raises(KeyError):
            weight_init_(weight, 'xxx')

    def test_conv1d_block(self):
        length = 2
        input = torch.rand(batch_size, in_channels, length).requires_grad_(True)
        block = conv1d_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=act,
            norm_type=norm_type
        )
        output = self.run_model(input, block)
        output_length = (length - kernel_size + 2 * padding // stride) + 1
        assert output.shape == (batch_size, out_channels, output_length)

    def test_conv2d_block(self):
        input = torch.rand(batch_size, in_channels, H, W).requires_grad_(True)
        for pad_type in ['zero', 'reflect', 'replication']:
            block = conv2d_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                pad_type=pad_type,
                activation=act,
                norm_type=norm_type
            )
            output = self.run_model(input, block)
            output_H = (H - kernel_size + 2 * padding // stride) + 1
            output_W = (W - kernel_size + 2 * padding // stride) + 1
            assert output.shape == (batch_size, out_channels, output_H, output_W)

    def test_deconv2d_block(self):
        input = torch.rand(batch_size, in_channels, H, W).requires_grad_(True)
        output_padding = 0
        block = deconv2d_block(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=output_padding,
            groups=1,
            activation=act,
            norm_type=norm_type
        )
        output = self.run_model(input, block)
        output_H = (H - 1) * stride + output_padding - 2 * padding + kernel_size
        output_W = (W - 1) * stride + output_padding - 2 * padding + kernel_size
        assert output.shape == (batch_size, out_channels, output_H, output_W)

    def test_fc_block(self):
        input = torch.rand(batch_size, in_channels).requires_grad_(True)
        for use_dropout in [True, False]:
            block = fc_block(
                in_channels,
                out_channels,
                activation=act,
                norm_type=norm_type,
                use_dropout=use_dropout,
                dropout_probability=0.5
            )
            output = self.run_model(input, block)
            assert output.shape == (batch_size, out_channels)

    def test_normed_linear(self):
        input = torch.rand(batch_size, in_channels).requires_grad_(True)
        block = normed_linear(in_channels, out_channels, scale=1)
        r = block.weight.norm(dim=None, keepdim=False) * block.weight.norm(dim=None, keepdim=False)
        assert r.item() < out_channels + 0.01
        assert r.item() > out_channels - 0.01
        output = self.run_model(input, block)
        assert output.shape == (batch_size, out_channels)

    def test_normed_conv2d(self):
        input = torch.rand(batch_size, in_channels, H, W).requires_grad_(True)
        block = normed_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            scale=1
        )
        r = block.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)[0, 0, 0, 0]
        assert r.item() < 1.01
        assert r.item() > 0.99
        output = self.run_model(input, block)
        output_H = (H - kernel_size + 2 * padding // stride) + 1
        output_W = (W - kernel_size + 2 * padding // stride) + 1
        assert output.shape == (batch_size, out_channels, output_H, output_W)

    def test_channel_shuffle(self):
        group_num = 2
        input = torch.rand(batch_size, in_channels, H, W).requires_grad_(True)
        channel_shuffle = ChannelShuffle(group_num)
        output = self.run_model(input, channel_shuffle)
        assert output.shape == input.shape

    def test_one_hot(self):
        M = 2
        N = 2
        max_num = 3
        input = torch.ones(M, N).long()
        output = one_hot(input, max_num, num_first=False)
        assert output.sum() == input.numel()
        assert output.shape == (M, N, max_num)
        output = one_hot(input, max_num, num_first=True)
        assert output.sum() == input.numel()
        assert output.shape == (max_num, M, N)
        with pytest.raises(RuntimeError):
            _ = one_hot(torch.arange(0, max_num), max_num - 1)

    def test_upsample(self):
        scale_factor = 2
        input = torch.rand(batch_size, in_channels, H, W).requires_grad_(True)
        model = NearestUpsample(scale_factor)
        output = self.run_model(input, model)
        assert output.shape == (batch_size, in_channels, 2 * H, 2 * W)
        model = BilinearUpsample(scale_factor)
        output = self.run_model(input, model)
        assert output.shape == (batch_size, in_channels, 2 * H, 2 * W)

    def test_binary_encode(self):
        input = torch.tensor([4])
        max_val = torch.tensor(8)
        output = binary_encode(input, max_val)
        assert torch.equal(output, torch.tensor([[0, 1, 0, 0]]))

    @pytest.mark.tmp
    def test_flatten(self):
        inputs = torch.randn(4, 3, 8, 8)
        model1 = NaiveFlatten()
        output1 = model1(inputs)
        assert output1.shape == (4, 3 * 8 * 8)
        model2 = NaiveFlatten(1, 2)
        output2 = model2(inputs)
        assert output2.shape == (4, 3 * 8, 8)
        model3 = NaiveFlatten(1, 3)
        output3 = model2(inputs)
        assert output1.shape == (4, 3 * 8 * 8)

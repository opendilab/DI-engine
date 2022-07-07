import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, kaiming_normal_, orthogonal_
from typing import Union, Tuple, List, Callable
from ding.compatibility import torch_ge_131

from .normalization import build_normalization


def weight_init_(weight: torch.Tensor, init_type: str = "xavier", activation: str = None) -> None:
    r"""
    Overview:
        Init weight according to the specified type.
    Arguments:
        - weight (:obj:`torch.Tensor`): the weight that needed to init
        - init_type (:obj:`str`): the type of init to implement, supports ["xavier", "kaiming", "orthogonal"]
        - activation (:obj:`str`): the activation function name, recommend that use only with \
            ['relu', 'leaky_relu'].
    """

    def xavier_init(weight, *args):
        xavier_normal_(weight)

    def kaiming_init(weight, activation):
        assert activation is not None
        if hasattr(activation, "negative_slope"):
            kaiming_normal_(weight, a=activation.negative_slope)
        else:
            kaiming_normal_(weight, a=0)

    def orthogonal_init(weight, *args):
        orthogonal_(weight)

    if init_type is None:
        return
    init_type_dict = {"xavier": xavier_init, "kaiming": kaiming_init, "orthogonal": orthogonal_init}
    if init_type in init_type_dict:
        init_type_dict[init_type](weight, activation)
    else:
        raise KeyError("Invalid Value in init type: {}".format(init_type))


def sequential_pack(layers: list) -> nn.Sequential:
    r"""
    Overview:
        Pack the layers in the input list to a `nn.Sequential` module.
        If there is a convolutional layer in module, an extra attribute `out_channels` will be added
        to the module and set to the out_channel of the conv layer.
    Arguments:
        - layers (:obj:`list`): the input list
    Returns:
        - seq (:obj:`nn.Sequential`): packed sequential container
    """
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in reversed(layers):
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def conv1d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        activation: nn.Module = None,
        norm_type: str = None
) -> nn.Sequential:
    r"""
    Overview:
        Create a 1-dim convlution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - kernel_size (:obj:`int`): Size of the convolving kernel
        - stride (:obj:`int`): Stride of the convolution
        - padding (:obj:`int`): Zero-padding added to both sides of the input
        - dilation (:obj:`int`): Spacing between kernel elements
        - groups (:obj:`int`): Number of blocked connections from input channels to output channels
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the 1 dim convlution layer

    .. note::

        Conv1d (https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d)
    """
    block = []
    block.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups))
    if norm_type is not None:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def conv2d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        pad_type: str = 'zero',
        activation: nn.Module = None,
        norm_type: str = None
) -> nn.Sequential:
    r"""
    Overview:
        Create a 2-dim convlution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - kernel_size (:obj:`int`): Size of the convolving kernel
        - stride (:obj:`int`): Stride of the convolution
        - padding (:obj:`int`): Zero-padding added to both sides of the input
        - dilation (:obj:`int`): Spacing between kernel elements
        - groups (:obj:`int`): Number of blocked connections from input channels to output channels
        - pad_type (:obj:`str`): the way to add padding, include ['zero', 'reflect', 'replicate'], default: None
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization, default set to None, now support ['BN', 'IN', 'SyncBN']
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the 2 dim convlution layer

    .. note::

        Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
    """
    block = []
    assert pad_type in ['zero', 'reflect', 'replication'], "invalid padding type: {}".format(pad_type)
    if pad_type == 'zero':
        pass
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
        padding = 0
    elif pad_type == 'replication':
        block.append(nn.ReplicationPad2d(padding))
        padding = 0
    block.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups)
    )
    if norm_type is not None:
        block.append(build_normalization(norm_type, dim=2)(out_channels))
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def deconv2d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        activation: int = None,
        norm_type: int = None
) -> nn.Sequential:
    r"""
    Overview:
        Create a 2-dim transopse convlution layer with activation and normalization
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - kernel_size (:obj:`int`): Size of the convolving kernel
        - stride (:obj:`int`): Stride of the convolution
        - padding (:obj:`int`): Zero-padding added to both sides of the input
        - pad_type (:obj:`str`): the way to add padding, include ['zero', 'reflect', 'replicate']
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the 2-dim \
            transpose convlution layer

    .. note::

        ConvTranspose2d (https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html)
    """
    block = []
    block.append(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups
        )
    )
    if norm_type is not None:
        block.append(build_normalization(norm_type, dim=2)(out_channels))
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def fc_block(
        in_channels: int,
        out_channels: int,
        activation: nn.Module = None,
        norm_type: str = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5
) -> nn.Sequential:
    r"""
    Overview:
        Create a fully-connected block with activation, normalization and dropout.
        Optional normalization can be done to the dim 1 (across the channels)
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
        - use_dropout (:obj:`bool`) : whether to use dropout in the fully-connected block
        - dropout_probability (:obj:`float`) : probability of an element to be zeroed in the dropout. Default: 0.5
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block

    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
    """
    block = []
    block.append(nn.Linear(in_channels, out_channels))
    if norm_type is not None:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


def normed_linear(in_features, out_features, bias: bool = True, device=None, dtype=None, scale=1.0):
    """
    nn.Linear but with normalized fan-in init
    """

    out = nn.Linear(in_features, out_features, bias)

    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if bias:
        out.bias.data.zero_()
    return out


def normed_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias: bool = True,
    padding_mode='zeros',
    device=None,
    dtype=None,
    scale=1
):
    """
    nn.Conv2d but with normalized fan-in init
    """
    out = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if bias:
        out.bias.data.zero_()
    return out


def MLP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    layer_num: int,
    layer_fn: Callable = None,
    activation: nn.Module = None,
    norm_type: str = None,
    use_dropout: bool = False,
    dropout_probability: float = 0.5
):
    r"""
    Overview:
        create a multi-layer perceptron using fully-connected blocks with activation, normalization and dropout,
        optional normalization can be done to the dim 1 (across the channels)
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - layer_num (:obj:`int`): Number of layers
        - layer_fn (:obj:`Callable`): layer function
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
        - use_dropout (:obj:`bool`): whether to use dropout in the fully-connected block
        - dropout_probability (:obj:`float`): probability of an element to be zeroed in the dropout. Default: 0.5
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block

    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class ChannelShuffle(nn.Module):
    r"""
    Overview:
        Apply channelShuffle to the input tensor
    Interface:
        forward

    .. note::

        You can see the original paper shuffle net in https://arxiv.org/abs/1707.01083
    """

    def __init__(self, group_num: int) -> None:
        r"""
        Overview:
            Init class ChannelShuffle
        Arguments:
            - group_num (:obj:`int`): the number of groups to exchange
        """
        super().__init__()
        self.group_num = group_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the upsampled input
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - x (:obj:`torch.Tensor`): the shuffled input tensor
        """
        b, c, h, w = x.shape
        g = self.group_num
        assert (c % g == 0)
        x = x.view(b, g, c // g, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        return x


def one_hot(val: torch.LongTensor, num: int, num_first: bool = False) -> torch.FloatTensor:
    r"""
    Overview:
        Convert a ``torch.LongTensor`` to one hot encoding.
        This implementation can be slightly faster than ``torch.nn.functional.one_hot``
    Arguments:
        - val (:obj:`torch.LongTensor`): each element contains the state to be encoded, the range should be [0, num-1]
        - num (:obj:`int`): number of states of the one hot encoding
        - num_first (:obj:`bool`): If ``num_first`` is False, the one hot encoding is added as the last; \
            Otherwise as the first dimension.
    Returns:
        - one_hot (:obj:`torch.FloatTensor`)
    Example:
        >>> one_hot(2*torch.ones([2,2]).long(),3)
        tensor([[[0., 0., 1.],
                 [0., 0., 1.]],
                [[0., 0., 1.],
                 [0., 0., 1.]]])
        >>> one_hot(2*torch.ones([2,2]).long(),3,num_first=True)
        tensor([[[0., 0.], [1., 0.]],
                [[0., 1.], [0., 0.]],
                [[1., 0.], [0., 1.]]])
    """
    assert (isinstance(val, torch.Tensor)), type(val)
    assert val.dtype == torch.long
    assert (len(val.shape) >= 1)
    old_shape = val.shape
    val_reshape = val.reshape(-1, 1)
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    # To remember the location where the original value is -1 in val.
    # If the value is -1, then it should be converted to all zeros encodings and
    # the corresponding entry in index_neg_one is 1, which is used to transform
    # the ret after the operation of ret.scatter_(1, val_reshape, 1) to their correct encodings bellowing
    index_neg_one = torch.eq(val_reshape, -1).float()
    if index_neg_one.sum() != 0:  # if -1 exists in val
        # convert the original value -1 to 0
        val_reshape = torch.where(
            val_reshape != -1, val_reshape,
            torch.zeros(val_reshape.shape, device=val.device).long()
        )
    try:
        ret.scatter_(1, val_reshape, 1)
        if index_neg_one.sum() != 0:  # if -1 exists in val
            ret = ret * (1 - index_neg_one)  # change -1's encoding from [1,0,...,0] to [0,0,...,0]
    except RuntimeError:
        raise RuntimeError('value: {}\nnum: {}\t:val_shape: {}\n'.format(val_reshape, num, val_reshape.shape))
    if num_first:
        return ret.permute(1, 0).reshape(num, *old_shape)
    else:
        return ret.reshape(*old_shape, num)


class NearestUpsample(nn.Module):
    r"""
    Overview:
        Upsamples the input to the given member varible scale_factor using mode nearest
    Interface:
        forward
    """

    def __init__(self, scale_factor: Union[float, List[float]]) -> None:
        r"""
        Overview:
            Init class NearestUpsample
        Arguments:
            - scale_factor (:obj:`Union[float, List[float]]`): multiplier for spatial size
        """
        super(NearestUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the upsampled input
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - upsample(:obj:`torch.Tensor`): the upsampled input tensor
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class BilinearUpsample(nn.Module):
    r"""
    Overview:
        Upsamples the input to the given member varible scale_factor using mode biliner
    Interface:
        forward
    """

    def __init__(self, scale_factor: Union[float, List[float]]) -> None:
        r"""
        Overview:
            Init class BilinearUpsample

        Arguments:
            - scale_factor (:obj:`Union[float, List[float]]`): multiplier for spatial size
        """
        super(BilinearUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the upsampled input
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - upsample(:obj:`torch.Tensor`): the upsampled input tensor
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


def binary_encode(y: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    r"""
    Overview:
        Convert elements in a tensor to its binary representation
    Arguments:
        - y (:obj:`torch.Tensor`): the tensor to be transferred into its binary representation
        - max_val (:obj:`torch.Tensor`): the max value of the elements in tensor
    Returns:
        - binary (:obj:`torch.Tensor`): the input tensor in its binary representation
    Example:
        >>> binary_encode(torch.tensor([3,2]),torch.tensor(8))
        tensor([[0, 0, 1, 1],[0, 0, 1, 0]])
    """
    assert (max_val > 0)
    x = y.clamp(0, max_val)
    L = int(math.log(max_val, 2)) + 1
    binary = []
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    for i in range(L):
        num = 1 << (L - i - 1)  # 2**(L-i-1)
        bit = torch.where(x >= num, one, zero)
        x -= bit * num
        binary.append(bit)
    return torch.stack(binary, dim=1)


class NoiseLinearLayer(nn.Module):
    r"""
    Overview:
        Linear layer with random noise.
    Interface:
        reset_noise, reset_parameters, forward
    """

    def __init__(self, in_channels: int, out_channels: int, sigma0: int = 0.4) -> None:
        super(NoiseLinearLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_channels))
        self.register_buffer("weight_eps", torch.empty(out_channels, in_channels))
        self.register_buffer("bias_eps", torch.empty(out_channels))
        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def _scale_noise(self, size: Union[int, Tuple]):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        r"""
        Overview:
            Reset noise settinngs in the layer.
        """
        is_cuda = self.weight_mu.is_cuda
        in_noise = self._scale_noise(self.in_channels).to(torch.device("cuda" if is_cuda else "cpu"))
        out_noise = self._scale_noise(self.out_channels).to(torch.device("cuda" if is_cuda else "cpu"))
        self.weight_eps = out_noise.ger(in_noise)
        self.bias_eps = out_noise

    def reset_parameters(self):
        r"""
        Overview:
            Reset parameters in the layer.
        """
        stdv = 1. / math.sqrt(self.in_channels)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)

        std_weight = self.sigma0 / math.sqrt(self.in_channels)
        self.weight_sigma.data.fill_(std_weight)
        std_bias = self.sigma0 / math.sqrt(self.out_channels)
        self.bias_sigma.data.fill_(std_bias)

    def forward(self, x: torch.Tensor):
        r"""
        Overview:
            Layer forward with noise.
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - output (:obj:`torch.Tensor`): the output with noise
        """
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_eps,
                self.bias_mu + self.bias_sigma * self.bias_eps,
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


def noise_block(
    in_channels: int,
    out_channels: int,
    activation: str = None,
    norm_type: str = None,
    use_dropout: bool = False,
    dropout_probability: float = 0.5,
    sigma0: float = 0.4
):
    r"""
    Overview:
        Create a fully-connected block with activation, normalization and dropout
        Optional normalization can be done to the dim 1 (across the channels)
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - activation (:obj:`str`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
        - use_dropout (:obj:`bool`) : whether to use dropout in the fully-connected block
        - dropout_probability (:obj:`float`) : probability of an element to be zeroed in the dropout. Default: 0.5
        - simga0 (:obj:`float`): the sigma0 is the defalut noise volumn when init NoiseLinearLayer
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block

    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
    """
    block = []
    block.append(NoiseLinearLayer(in_channels, out_channels, sigma0=sigma0))
    if norm_type is not None:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class NaiveFlatten(nn.Module):

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(NaiveFlatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.end_dim != -1:
            return x.view(*x.shape[:self.start_dim], -1, *x.shape[self.end_dim + 1:])
        else:
            return x.view(*x.shape[:self.start_dim], -1)


if torch_ge_131():
    Flatten = nn.Flatten
else:
    Flatten = NaiveFlatten

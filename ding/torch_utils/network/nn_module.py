import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, kaiming_normal_, orthogonal_
from typing import Union, Tuple, List, Callable
from ding.compatibility import torch_ge_131

from .normalization import build_normalization


def weight_init_(weight: torch.Tensor, init_type: str = "xavier", activation: str = None) -> None:
    """
    Overview:
        Initialize weight according to the specified type.
    Arguments:
        - weight (:obj:`torch.Tensor`): The weight that needs to be initialized.
        - init_type (:obj:`str`, optional): The type of initialization to implement, \
            supports ["xavier", "kaiming", "orthogonal"].
        - activation (:obj:`str`, optional): The activation function name. Recommended to use only with \
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


def sequential_pack(layers: List[nn.Module]) -> nn.Sequential:
    """
    Overview:
        Pack the layers in the input list to a `nn.Sequential` module.
        If there is a convolutional layer in module, an extra attribute `out_channels` will be added
        to the module and set to the out_channel of the conv layer.
    Arguments:
        - layers (:obj:`List[nn.Module]`): The input list of layers.
    Returns:
        - seq (:obj:`nn.Sequential`): Packed sequential container.
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
    """
    Overview:
        Create a 1-dimensional convolution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - kernel_size (:obj:`int`): Size of the convolving kernel.
        - stride (:obj:`int`, optional): Stride of the convolution. Default is 1.
        - padding (:obj:`int`, optional): Zero-padding added to both sides of the input. Default is 0.
        - dilation (:obj:`int`, optional): Spacing between kernel elements. Default is 1.
        - groups (:obj:`int`, optional): Number of blocked connections from input channels to output channels. \
            Default is 1.
        - activation (:obj:`nn.Module`, optional): The optional activation function.
        - norm_type (:obj:`str`, optional): Type of the normalization.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the 1-dimensional \
            convolution layer.

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
        norm_type: str = None,
        num_groups_for_gn: int = 1,
        bias: bool = True
) -> nn.Sequential:
    """
    Overview:
        Create a 2-dimensional convolution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - kernel_size (:obj:`int`): Size of the convolving kernel.
        - stride (:obj:`int`, optional): Stride of the convolution. Default is 1.
        - padding (:obj:`int`, optional): Zero-padding added to both sides of the input. Default is 0.
        - dilation (:obj:`int`): Spacing between kernel elements.
        - groups (:obj:`int`, optional): Number of blocked connections from input channels to output channels. \
            Default is 1.
        - pad_type (:obj:`str`, optional): The way to add padding, include ['zero', 'reflect', 'replicate']. \
            Default is 'zero'.
        - activation (:obj:`nn.Module`): the optional activation function.
        - norm_type (:obj:`str`): The type of the normalization, now support ['BN', 'LN', 'IN', 'GN', 'SyncBN'], \
            default set to None, which means no normalization.
        - num_groups_for_gn (:obj:`int`): Number of groups for GroupNorm.
        - bias (:obj:`bool`): whether to add a learnable bias to the nn.Conv2d. Default is True.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the 2-dimensional \
            convolution layer.

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
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    )
    if norm_type is not None:
        if norm_type == 'LN':
            # LN is implemented as GroupNorm with 1 group.
            block.append(nn.GroupNorm(1, out_channels))
        elif norm_type == 'GN':
            block.append(nn.GroupNorm(num_groups_for_gn, out_channels))
        elif norm_type in ['BN', 'IN', 'SyncBN']:
            block.append(build_normalization(norm_type, dim=2)(out_channels))
        else:
            raise KeyError(
                "Invalid value in norm_type: {}. The valid norm_type are "
                "BN, LN, IN, GN and SyncBN.".format(norm_type)
            )

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
    """
    Overview:
        Create a 2-dimensional transpose convolution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - kernel_size (:obj:`int`): Size of the convolving kernel.
        - stride (:obj:`int`, optional): Stride of the convolution. Default is 1.
        - padding (:obj:`int`, optional): Zero-padding added to both sides of the input. Default is 0.
        - output_padding (:obj:`int`, optional): Additional size added to one side of the output shape. Default is 0.
        - groups (:obj:`int`, optional): Number of blocked connections from input channels to output channels. \
            Default is 1.
        - activation (:obj:`int`, optional): The optional activation function.
        - norm_type (:obj:`int`, optional): Type of the normalization.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the 2-dimensional \
            transpose convolution layer.

    .. note::

        ConvTranspose2d (https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html)
    """
    block = [
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups
        )
    ]
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
    """
    Overview:
        Create a fully-connected block with activation, normalization, and dropout.
        Optional normalization can be done to the dim 1 (across the channels).
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - activation (:obj:`nn.Module`, optional): The optional activation function.
        - norm_type (:obj:`str`, optional): Type of the normalization.
        - use_dropout (:obj:`bool`, optional): Whether to use dropout in the fully-connected block. Default is False.
        - dropout_probability (:obj:`float`, optional): Probability of an element to be zeroed in the dropout. \
            Default is 0.5.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the fully-connected block.

    .. note::

        You can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html).
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


def normed_linear(
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        scale: float = 1.0
) -> nn.Linear:
    """
    Overview:
        Create a nn.Linear module but with normalized fan-in init.
    Arguments:
        - in_features (:obj:`int`): Number of features in the input tensor.
        - out_features (:obj:`int`): Number of features in the output tensor.
        - bias (:obj:`bool`, optional): Whether to add a learnable bias to the nn.Linear. Default is True.
        - device (:obj:`torch.device`, optional): The device to put the created module on. Default is None.
        - dtype (:obj:`torch.dtype`, optional): The desired data type of created module. Default is None.
        - scale (:obj:`float`, optional): The scale factor for initialization. Default is 1.0.
    Returns:
        - out (:obj:`nn.Linear`): A nn.Linear module with normalized fan-in init.
    """

    out = nn.Linear(in_features, out_features, bias)

    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if bias:
        out.bias.data.zero_()
    return out


def normed_conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        scale: float = 1
) -> nn.Conv2d:
    """
    Overview:
        Create a nn.Conv2d module but with normalized fan-in init.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - kernel_size (:obj:`Union[int, Tuple[int, int]]`): Size of the convolving kernel.
        - stride (:obj:`Union[int, Tuple[int, int]]`, optional): Stride of the convolution. Default is 1.
        - padding (:obj:`Union[int, Tuple[int, int]]`, optional): Zero-padding added to both sides of the input. \
            Default is 0.
        - dilation (:`Union[int, Tuple[int, int]]`, optional): Spacing between kernel elements. Default is 1.
        - groups (:obj:`int`, optional): Number of blocked connections from input channels to output channels. \
            Default is 1.
        - bias (:obj:`bool`, optional): Whether to add a learnable bias to the nn.Conv2d. Default is True.
        - padding_mode (:obj:`str`, optional): The type of padding algorithm to use. Default is 'zeros'.
        - device (:obj:`torch.device`, optional): The device to put the created module on. Default is None.
        - dtype (:obj:`torch.dtype`, optional): The desired data type of created module. Default is None.
        - scale (:obj:`float`, optional): The scale factor for initialization. Default is 1.
    Returns:
        - out (:obj:`nn.Conv2d`): A nn.Conv2d module with normalized fan-in init.
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
    dropout_probability: float = 0.5,
    output_activation: bool = True,
    output_norm: bool = True,
    last_linear_layer_init_zero: bool = False
):
    """
    Overview:
        Create a multi-layer perceptron using fully-connected blocks with activation, normalization, and dropout,
        optional normalization can be done to the dim 1 (across the channels).
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - layer_num (:obj:`int`): Number of layers.
        - layer_fn (:obj:`Callable`, optional): Layer function.
        - activation (:obj:`nn.Module`, optional): The optional activation function.
        - norm_type (:obj:`str`, optional): The type of the normalization.
        - use_dropout (:obj:`bool`, optional): Whether to use dropout in the fully-connected block. Default is False.
        - dropout_probability (:obj:`float`, optional): Probability of an element to be zeroed in the dropout. \
            Default is 0.5.
        - output_activation (:obj:`bool`, optional): Whether to use activation in the output layer. If True, \
            we use the same activation as front layers. Default is True.
        - output_norm (:obj:`bool`, optional): Whether to use normalization in the output layer. If True, \
            we use the same normalization as front layers. Default is True.
        - last_linear_layer_init_zero (:obj:`bool`, optional): Whether to use zero initializations for the last \
            linear layer (including w and b), which can provide stable zero outputs in the beginning, \
            usually used in the policy network in RL settings.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the multi-layer perceptron.

    .. note::
        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html).
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    # The last layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))
    """
    In the final layer of a neural network, whether to use normalization and activation are typically determined
    based on user specifications. These specifications depend on the problem at hand and the desired properties of
    the model's output.
    """
    if output_norm is True:
        # The last layer uses the same norm as front layers.
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
    if output_activation is True:
        # The last layer uses the same activation as front layers.
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    if last_linear_layer_init_zero:
        # Locate the last linear layer and initialize its weights and biases to 0.
        for _, layer in enumerate(reversed(block)):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return sequential_pack(block)


class ChannelShuffle(nn.Module):
    """
    Overview:
        Apply channel shuffle to the input tensor. For more details about the channel shuffle,
        please refer to the 'ShuffleNet' paper: https://arxiv.org/abs/1707.01083
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, group_num: int) -> None:
        """
        Overview:
            Initialize the ChannelShuffle class.
        Arguments:
            - group_num (:obj:`int`): The number of groups to exchange.
        """
        super().__init__()
        self.group_num = group_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass through the ChannelShuffle module.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The shuffled input tensor.
        """
        b, c, h, w = x.shape
        g = self.group_num
        assert (c % g == 0)
        x = x.view(b, g, c // g, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        return x


def one_hot(val: torch.LongTensor, num: int, num_first: bool = False) -> torch.FloatTensor:
    """
    Overview:
        Convert a torch.LongTensor to one-hot encoding. This implementation can be slightly faster than
        ``torch.nn.functional.one_hot``.
    Arguments:
        - val (:obj:`torch.LongTensor`): Each element contains the state to be encoded, the range should be [0, num-1]
        - num (:obj:`int`): Number of states of the one-hot encoding
        - num_first (:obj:`bool`, optional): If False, the one-hot encoding is added as the last dimension; otherwise, \
            it is added as the first dimension. Default is False.
    Returns:
        - one_hot (:obj:`torch.FloatTensor`): The one-hot encoded tensor.
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
    """
    Overview:
        This module upsamples the input to the given scale_factor using the nearest mode.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, scale_factor: Union[float, List[float]]) -> None:
        """
        Overview:
            Initialize the NearestUpsample class.
        Arguments:
            - scale_factor (:obj:`Union[float, List[float]]`): The multiplier for the spatial size.
        """
        super(NearestUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
         Overview:
             Return the upsampled input tensor.
         Arguments:
             - x (:obj:`torch.Tensor`): The input tensor.
         Returns:
             - upsample(:obj:`torch.Tensor`): The upsampled input tensor.
         """
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class BilinearUpsample(nn.Module):
    """
    Overview:
        This module upsamples the input to the given scale_factor using the bilinear mode.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, scale_factor: Union[float, List[float]]) -> None:
        """
        Overview:
            Initialize the BilinearUpsample class.
        Arguments:
            - scale_factor (:obj:`Union[float, List[float]]`): The multiplier for the spatial size.
        """
        super(BilinearUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the upsampled input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - upsample(:obj:`torch.Tensor`): The upsampled input tensor.
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


def binary_encode(y: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Convert elements in a tensor to its binary representation.
    Arguments:
        - y (:obj:`torch.Tensor`): The tensor to be converted into its binary representation.
        - max_val (:obj:`torch.Tensor`): The maximum value of the elements in the tensor.
    Returns:
        - binary (:obj:`torch.Tensor`): The input tensor in its binary representation.
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
    """
    Overview:
        This is a linear layer with random noise.
    Interfaces:
        ``__init__``, ``reset_noise``, ``reset_parameters``, ``forward``
    """

    def __init__(self, in_channels: int, out_channels: int, sigma0: int = 0.4) -> None:
        """
        Overview:
            Initialize the NoiseLinearLayer class. The 'enable_noise' attribute enables external control over whether \
            noise is applied.
            - If enable_noise is True, the layer adds noise even if the module is in evaluation mode.
            - If enable_noise is False, no noise is added regardless of self.training.
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor.
            - out_channels (:obj:`int`): Number of channels in the output tensor.
            - sigma0 (:obj:`int`, optional): Default noise volume when initializing NoiseLinearLayer. \
                Default is 0.4.
        """
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
        self.enable_noise = False
        self.reset_parameters()
        self.reset_noise()

    def _scale_noise(self, size: Union[int, Tuple]):
        """
        Overview:
            Scale the noise.
        Arguments:
            - size (:obj:`Union[int, Tuple]`): The size of the noise.
        """

        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        """
         Overview:
             Reset the noise settings in the layer.
         """
        is_cuda = self.weight_mu.is_cuda
        in_noise = self._scale_noise(self.in_channels).to(torch.device("cuda" if is_cuda else "cpu"))
        out_noise = self._scale_noise(self.out_channels).to(torch.device("cuda" if is_cuda else "cpu"))
        self.weight_eps = out_noise.ger(in_noise)
        self.bias_eps = out_noise

    def reset_parameters(self):
        """
        Overview:
            Reset the parameters in the layer.
        """
        stdv = 1. / math.sqrt(self.in_channels)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)

        std_weight = self.sigma0 / math.sqrt(self.in_channels)
        self.weight_sigma.data.fill_(std_weight)
        std_bias = self.sigma0 / math.sqrt(self.out_channels)
        self.bias_sigma.data.fill_(std_bias)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            Perform the forward pass with noise.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - output (:obj:`torch.Tensor`): The output tensor with noise.
        """
        # Determine whether to add noise:
        if self.enable_noise:
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
    """
    Overview:
        Create a fully-connected noise layer with activation, normalization, and dropout.
        Optional normalization can be done to the dim 1 (across the channels).
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - activation (:obj:`str`, optional): The optional activation function. Default is None.
        - norm_type (:obj:`str`, optional): Type of normalization. Default is None.
        - use_dropout (:obj:`bool`, optional): Whether to use dropout in the fully-connected block.
        - dropout_probability (:obj:`float`, optional): Probability of an element to be zeroed in the dropout. \
            Default is 0.5.
        - sigma0 (:obj:`float`, optional): The sigma0 is the default noise volume when initializing NoiseLinearLayer. \
            Default is 0.4.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the fully-connected block.
    """
    block = [NoiseLinearLayer(in_channels, out_channels, sigma0=sigma0)]
    if norm_type is not None:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class NaiveFlatten(nn.Module):
    """
    Overview:
        This module is a naive implementation of the flatten operation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """
        Overview:
            Initialize the NaiveFlatten class.
        Arguments:
            - start_dim (:obj:`int`, optional): The first dimension to flatten. Default is 1.
            - end_dim (:obj:`int`, optional): The last dimension to flatten. Default is -1.
        """
        super(NaiveFlatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Perform the flatten operation on the input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - output (:obj:`torch.Tensor`): The flattened output tensor.
        """
        if self.end_dim != -1:
            return x.view(*x.shape[:self.start_dim], -1, *x.shape[self.end_dim + 1:])
        else:
            return x.view(*x.shape[:self.start_dim], -1)


if torch_ge_131():
    Flatten = nn.Flatten
else:
    Flatten = NaiveFlatten

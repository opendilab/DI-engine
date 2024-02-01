"""
This implementation of ResNet is a bit modification version of `https://github.com/rwightman/pytorch-image-models.git`
"""
from typing import List, Callable, Optional, Tuple, Type, Dict, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_module import Flatten


def to_2tuple(item: int) -> tuple:
    """
    Overview:
        Convert a scalar to a 2-tuple or return the item if it's not a scalar.
    Arguments:
        - item (:obj:`int`): An item to be converted to a 2-tuple.
    Returns:
        - (:obj:`tuple`): A 2-tuple of the item.
    """
    if np.isscalar(item):
        return (item, item)
    else:
        return item


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int) -> int:
    """
    Overview:
        Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution.
    Arguments:
        - x (:obj:`int`): The size of the input.
        - k (:obj:`int`): The size of the kernel.
        - s (:obj:`int`): The stride of the convolution.
        - d (:obj:`int`): The dilation of the convolution.
    Returns:
        - (:obj:`int`): The size of the padding.
    """
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    """
    Overview:
        Dynamically pad input x with 'SAME' padding for conv with specified args.
    Arguments:
        - x (:obj:`Tensor`): The input tensor.
        - k (:obj:`List[int]`): The size of the kernel.
        - s (:obj:`List[int]`): The stride of the convolution.
        - d (:obj:`List[int]`): The dilation of the convolution.
        - value (:obj:`float`): Value to fill the padding.
    Returns:
        - (:obj:`Tensor`): The padded tensor.
    """
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def avg_pool2d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True
):
    """
    Overview:
        Apply average pooling with 'SAME' padding on the input tensor.
    Arguments:
        - x (:obj:`Tensor`): The input tensor.
        - kernel_size (:obj:`List[int]`): The size of the kernel.
        - stride (:obj:`List[int]`): The stride of the convolution.
        - padding (:obj:`List[int]`): The size of the padding.
        - ceil_mode (:obj:`bool`): When True, will use ceil instead of floor to compute the output shape.
        - count_include_pad (:obj:`bool`): When True, will include the zero-padding in the averaging calculation.
    Returns:
        - (:obj:`Tensor`): The tensor after average pooling.
    """
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """
    Overview:
        Tensorflow-like 'SAME' wrapper for 2D average pooling.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            kernel_size: int,
            stride: Optional[Tuple[int, int]] = None,
            padding: int = 0,
            ceil_mode: bool = False,
            count_include_pad: bool = True
    ) -> None:
        """
        Overview:
            Initialize the AvgPool2dSame with given arguments.
        Arguments:
            - kernel_size (:obj:`int`): The size of the window to take an average over.
            - stride (:obj:`Optional[Tuple[int, int]]`): The stride of the window. If None, default to kernel_size.
            - padding (:obj:`int`): Implicit zero padding to be added on both sides.
            - ceil_mode (:obj:`bool`): When True, will use `ceil` instead of `floor` to compute the output shape.
            - count_include_pad (:obj:`bool`): When True, will include the zero-padding in the averaging calculation.
        """
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the AvgPool2dSame.
        Argument:
            - x (:obj:`torch.Tensor`): Input tensor.
        Returns:
            - (:obj:`torch.Tensor`): Output tensor after average pooling.
        """
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


def _create_pool(num_features: int,
                 num_classes: int,
                 pool_type: str = 'avg',
                 use_conv: bool = False) -> Tuple[nn.Module, int]:
    """
    Overview:
        Create a global pooling layer based on the given arguments.
    Arguments:
        - num_features (:obj:`int`): Number of input features.
        - num_classes (:obj:`int`): Number of output classes.
        - pool_type (:obj:`str`): Type of the pooling operation. Defaults to 'avg'.
        - use_conv (:obj:`bool`): Whether to use convolutional layer after pooling. Defaults to False.
    Returns:
        - (:obj:`Tuple[nn.Module, int]`): The created global pooling layer and the number of pooled features.
    """
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv, \
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    assert flatten_in_pool
    global_pool = nn.AdaptiveAvgPool2d(1)
    num_pooled_features = num_features * 1
    return global_pool, num_pooled_features


def _create_fc(num_features: int, num_classes: int, use_conv: bool = False) -> nn.Module:
    """
    Overview:
        Create a fully connected layer based on the given arguments.
    Arguments:
        - num_features (:obj:`int`): Number of input features.
        - num_classes (:obj:`int`): Number of output classes.
        - use_conv (:obj:`bool`): Whether to use convolutional layer. Defaults to False.
    Returns:
        - (:obj:`nn.Module`): The created fully connected layer.
    """
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        # use nn.Linear for simplification
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


def create_classifier(num_features: int,
                      num_classes: int,
                      pool_type: str = 'avg',
                      use_conv: bool = False) -> Tuple[nn.Module, nn.Module]:
    """
    Overview:
        Create a classifier with global pooling layer and fully connected layer.
    Arguments:
        - num_features (:obj:`int`): The number of features.
        - num_classes (:obj:`int`): The number of classes for the final classification.
        - pool_type (:obj:`str`): The type of pooling to use; 'avg' for Average Pooling.
        - use_conv (:obj:`bool`): Whether to use convolution or not.
    Returns:
        - global_pool (:obj:`nn.Module`): The created global pooling layer.
        - fc (:obj:`nn.Module`): The created fully connected layer.
    """
    assert pool_type == 'avg'
    global_pool, num_pooled_features = _create_pool(num_features, num_classes, pool_type, use_conv=use_conv)
    fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
    return global_pool, fc


class ClassifierHead(nn.Module):
    """
    Overview:
        Classifier head with configurable global pooling and dropout.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            in_chs: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False
    ) -> None:
        """
        Overview:
            Initialize the ClassifierHead with given arguments.
        Arguments:
            - in_chs (:obj:`int`): Number of input channels.
            - num_classes (:obj:`int`): Number of classes for the final classification.
            - pool_type (:obj:`str`): The type of pooling to use; 'avg' for Average Pooling.
            - drop_rate (:obj:`float`): The dropout rate.
            - use_conv (:obj:`bool`): Whether to use convolution or not.
        """
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten = Flatten(1) if use_conv and pool_type else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the ClassifierHead.
        Argument:
            - x (:obj:`torch.Tensor`): Input tensor.
        Returns:
            - (:obj:`torch.Tensor`): Output tensor after classification.
        """
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        x = self.flatten(x)
        return x


def create_attn(layer: nn.Module, plane: int) -> None:
    """
    Overview:
        Create an attention mechanism.
    Arguments:
        - layer (:obj:`nn.Module`): The layer where the attention is to be applied.
        - plane (:obj:`int`): The plane on which the attention is to be applied.
    Returns:
        - None
    """
    return None


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    """
    Overview:
        Compute the padding based on the kernel size, stride and dilation.
    Arguments:
        - kernel_size (:obj:`int`): The size of the kernel.
        - stride (:obj:`int`): The stride of the convolution.
        - dilation (:obj:`int`): The dilation factor.
    Returns:
        - padding (:obj:`int`): The computed padding.
    """
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock(nn.Module):
    """
    Overview:
        The basic building block for models like ResNet. This class extends pytorch's Module class.
        It represents a standard block of layers including two convolutions, batch normalization,
        an optional attention mechanism, and activation functions.
    Interfaces:
        ``__init__``, ``forward``, ``zero_init_last_bn``
    Properties:
        - expansion (:obj:int): Specifies the expansion factor for the planes of the conv layers.
    """
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Callable = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: int = None,
            act_layer: Callable = nn.ReLU,
            norm_layer: Callable = nn.BatchNorm2d,
            attn_layer: Callable = None,
            aa_layer: Callable = None,
            drop_block: Callable = None,
            drop_path: Callable = None
    ) -> None:
        """
        Overview:
            Initialize the BasicBlock with given parameters.
        Arguments:
            - inplanes (:obj:`int`): Number of input channels.
            - planes (:obj:`int`): Number of output channels.
            - stride (:obj:`int`): The stride of the convolutional layer.
            - downsample (:obj:`Callable`): Function for downsampling the inputs.
            - cardinality (:obj:`int`): Group size for grouped convolution.
            - base_width (:obj:`int`): Base width of the convolutions.
            - reduce_first (:obj:`int`): Reduction factor for first convolution of each block.
            - dilation (:obj:`int`): Spacing between kernel points.
            - first_dilation (:obj:`int`): First dilation value.
            - act_layer (:obj:`Callable`): Function for activation layer.
            - norm_layer (:obj:`Callable`): Function for normalization layer.
            - attn_layer (:obj:`Callable`): Function for attention layer.
            - aa_layer (:obj:`Callable`): Function for anti-aliasing layer.
            - drop_block (:obj:`Callable`): Method for dropping block.
            - drop_path (:obj:`Callable`): Method for dropping path.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes,
            first_planes,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False
        )
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.aa = aa_layer(channels=first_planes, stride=stride) if use_aa else None

        self.conv2 = nn.Conv2d(first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self) -> None:
        """
        Overview:
            Initialize the batch normalization layer with zeros.
        """
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Defines the computation performed at every call.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - output (:obj:`torch.Tensor`): The output tensor after passing through the BasicBlock.
        """
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    """
    Overview:
        The Bottleneck class is a basic block used to build ResNet networks. It is a part of the PyTorch's
        implementation of ResNet. This block is designed with several layers including a convolutional layer,
        normalization layer, activation layer, attention layer, anti-aliasing layer, and a dropout layer.
    Interfaces:
        ``__init__``, ``forward``, ``zero_init_last_bn``
    Properties:
        expansion, inplanes, planes, stride, downsample, cardinality, base_width, reduce_first, dilation, \
        first_dilation, act_layer, norm_layer, attn_layer, aa_layer, drop_block, drop_path

    """
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Callable = None,
            drop_path: Callable = None
    ) -> None:
        """
        Overview:
            Initialize the Bottleneck class with various parameters.

        Arguments:
            - inplanes (:obj:`int`): The number of input planes.
            - planes (:obj:`int`): The number of output planes.
            - stride (:obj:`int`, optional): The stride size, defaults to 1.
            - downsample (:obj:`nn.Module`, optional): The downsample method, defaults to None.
            - cardinality (:obj:`int`, optional): The size of the group convolutions, defaults to 1.
            - base_width (:obj:`int`, optional): The base width, defaults to 64.
            - reduce_first (:obj:`int`, optional): The first reduction factor, defaults to 1.
            - dilation (:obj:`int`, optional): The dilation factor, defaults to 1.
            - first_dilation (:obj:`int`, optional): The first dilation factor, defaults to None.
            - act_layer (:obj:`Type[nn.Module]`, optional): The activation layer type, defaults to nn.ReLU.
            - norm_layer (:obj:`Type[nn.Module]`, optional): The normalization layer type, defaults to nn.BatchNorm2d.
            - attn_layer (:obj:`Type[nn.Module]`, optional): The attention layer type, defaults to None.
            - aa_layer (:obj:`Type[nn.Module]`, optional): The anti-aliasing layer type, defaults to None.
            - drop_block (:obj:`Callable`): The dropout block, defaults to None.
            - drop_path (:obj:`Callable`): The drop path, defaults to None.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            width,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False
        )
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self) -> None:
        """
        Overview:
            Initialize the last batch normalization layer with zero.
        """
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Defines the computation performed at every call.
        Arguments:
            - x (:obj:`Tensor`): The input tensor.
        Returns:
            - x (:obj:`Tensor`): The output tensor resulting from the computation.
        """
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: int = None,
        norm_layer: Type[nn.Module] = None
) -> nn.Sequential:
    """
    Overview:
        Create a sequential module for downsampling that includes a convolution layer and a normalization layer.
    Arguments:
        - in_channels (:obj:`int`): The number of input channels.
        - out_channels (:obj:`int`): The number of output channels.
        - kernel_size (:obj:`int`): The size of the kernel.
        - stride (:obj:`int`, optional): The stride size, defaults to 1.
        - dilation (:obj:`int`, optional): The dilation factor, defaults to 1.
        - first_dilation (:obj:`int`, optional): The first dilation factor, defaults to None.
        - norm_layer (:obj:`Type[nn.Module]`, optional): The normalization layer type, defaults to nn.BatchNorm2d.
    Returns:
        - nn.Sequential: A sequence of layers performing downsampling through convolution.
    """
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False
            ),
            norm_layer(out_channels)
        ]
    )


def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: int = None,
        norm_layer: Type[nn.Module] = None
) -> nn.Sequential:
    """
    Overview:
        Create a sequential module for downsampling that includes an average pooling layer, a convolution layer,
        and a normalization layer.
    Arguments:
        - in_channels (:obj:`int`): The number of input channels.
        - out_channels (:obj:`int`): The number of output channels.
        - kernel_size (:obj:`int`): The size of the kernel.
        - stride (:obj:`int`, optional): The stride size, defaults to 1.
        - dilation (:obj:`int`, optional): The dilation factor, defaults to 1.
        - first_dilation (:obj:`int`, optional): The first dilation factor, defaults to None.
        - norm_layer (:obj:`Type[nn.Module]`, optional): The normalization layer type, defaults to nn.BatchNorm2d.
    Returns:
        - nn.Sequential: A sequence of layers performing downsampling through average pooling.
    """
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[pool,
          nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
          norm_layer(out_channels)]
    )


def drop_blocks(drop_block_rate: float = 0.) -> List[None]:
    """
    Overview:
        Generate a list of None values based on the drop block rate.
    Arguments:
        - drop_block_rate (:obj:`float`, optional): The drop block rate, defaults to 0.
    Returns:
        - List[None]: A list of None values.
    """
    assert drop_block_rate == 0., drop_block_rate
    return [None for _ in range(4)]


def make_blocks(
        block_fn: Type[nn.Module],
        channels: List[int],
        block_repeats: List[int],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Union[int, str]]]]:
    """
    Overview:
        Create a list of blocks for the network, with each block having a given number of repeats. Also, create a
        feature info list that contains information about the output of each block.
    Arguments:
        - block_fn (:obj:`Type[nn.Module]`): The type of block to use.
        - channels (:obj:`List[int]`): The list of output channels for each block.
        - block_repeats (:obj:`List[int]`): The list of number of repeats for each block.
        - inplanes (:obj:`int`): The number of input planes.
        - reduce_first (:obj:`int`, optional): The first reduction factor, defaults to 1.
        - output_stride (:obj:`int`, optional): The total stride of the network, defaults to 32.
        - down_kernel_size (:obj:`int`, optional): The size of the downsample kernel, defaults to 1.
        - avg_down (:obj:`bool`, optional): Whether to use average pooling for downsampling, defaults to False.
        - drop_block_rate (:obj:`float`, optional): The drop block rate, defaults to 0.
        - drop_path_rate (:obj:`float`, optional): The drop path rate, defaults to 0.
    Returns:
        - Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Union[int, str]]]]: \
            A tuple that includes a list of blocks for the network and a feature info list.
    """
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer')
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(
                block_fn(
                    inplanes, planes, stride, downsample, first_dilation=prev_dilation, drop_path=None, **block_kwargs
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):
    """
    Overview:
        Implements ResNet, ResNeXt, SE-ResNeXt, and SENet models. This implementation supports various modifications
        based on the v1c, v1d, v1e, and v1s variants included in the MXNet Gluon ResNetV1b model. For more details
        about the variants and options, please refer to the 'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187.
    Interfaces:
        ``__init__``, ``forward``, ``zero_init_last_bn``, ``get_classifier``
    """

    def __init__(
            self,
            block: nn.Module,
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            output_stride: int = 32,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            act_layer: nn.Module = nn.ReLU,
            norm_layer: nn.Module = nn.BatchNorm2d,
            aa_layer: Optional[nn.Module] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            drop_block_rate: float = 0.0,
            global_pool: str = 'avg',
            zero_init_last_bn: bool = True,
            block_args: Optional[dict] = None
    ) -> None:
        """
        Overview:
            Initialize the ResNet model with given block, layers and other configuration options.
        Arguments:
            - block (:obj:`nn.Module`): Class for the residual block.
            - layers (:obj:`List[int]`): Numbers of layers in each block.
            - num_classes (:obj:`int`, optional): Number of classification classes. Default is 1000.
            - in_chans (:obj:`int`, optional): Number of input (color) channels. Default is 3.
            - cardinality (:obj:`int`, optional): Number of convolution groups for 3x3 conv in Bottleneck. Default is 1.
            - base_width (:obj:`int`, optional): Factor determining bottleneck channels. Default is 64.
            - stem_width (:obj:`int`, optional): Number of channels in stem convolutions. Default is 64.
            - stem_type (:obj:`str`, optional): The type of stem. Default is ''.
            - replace_stem_pool (:obj:`bool`, optional): Whether to replace stem pooling. Default is False.
            - output_stride (:obj:`int`, optional): Output stride of the network. Default is 32.
            - block_reduce_first (:obj:`int`, optional): Reduction factor for first convolution output width of \
                residual blocks. Default is 1.
            - down_kernel_size (:obj:`int`, optional): Kernel size of residual block downsampling path. Default is 1.
            - avg_down (:obj:`bool`, optional): Whether to use average pooling for projection skip connection between
                stages/downsample. Default is False.
            - act_layer (:obj:`nn.Module`, optional): Activation layer. Default is nn.ReLU.
            - norm_layer (:obj:`nn.Module`, optional): Normalization layer. Default is nn.BatchNorm2d.
            - aa_layer (:obj:`Optional[nn.Module]`, optional): Anti-aliasing layer. Default is None.
            - drop_rate (:obj:`float`, optional): Dropout probability before classifier, for training. Default is 0.0.
            - drop_path_rate (:obj:`float`, optional): Drop path rate. Default is 0.0.
            - drop_block_rate (:obj:`float`, optional): Drop block rate. Default is 0.0.
            - global_pool (:obj:`str`, optional): Global pooling type. Default is 'avg'.
            - zero_init_last_bn (:obj:`bool`, optional): Whether to initialize last batch normalization with zero. \
                Default is True.
            - block_args (:obj:`Optional[dict]`, optional): Additional arguments for block. Default is None.
        """
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(ResNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                    norm_layer(stem_chs[0]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                    norm_layer(stem_chs[1]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)
                ]
            )
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem Pooling
        if replace_stem_pool:
            self.maxpool = nn.Sequential(
                *filter(
                    None, [
                        nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                        aa_layer(channels=inplanes, stride=2) if aa_layer else None,
                        norm_layer(inplanes),
                        act_layer(inplace=True)
                    ]
                )
            )
        else:
            if aa_layer is not None:
                self.maxpool = nn.Sequential(
                    *[nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                      aa_layer(channels=inplanes, stride=2)]
                )
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last_bn=zero_init_last_bn)

    def init_weights(self, zero_init_last_bn: bool = True) -> None:
        """
        Overview:
            Initialize the weights in the model.
        Arguments:
            - zero_init_last_bn (:obj:`bool`, optional): Whether to initialize last batch normalization with zero.
                Default is True.
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self) -> nn.Module:
        """
        Overview:
            Get the classifier module from the model.
        Returns:
            - classifier (:obj:`nn.Module`): The classifier module in the model.
        """
        return self.fc

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg') -> None:
        """
        Overview:
            Reset the classifier with a new number of classes and pooling type.
        Arguments:
            - num_classes (:obj:`int`): New number of classification classes.
            - global_pool (:obj:`str`, optional): New global pooling type. Default is 'avg'.
        """
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass through the feature layers of the model.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor after passing through feature layers.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Full forward pass through the model.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor after passing through the model.
        """
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def resnet18() -> nn.Module:
    """
    Overview:
        Creates a ResNet18 model.
    Returns:
        - model (:obj:`nn.Module`): ResNet18 model.
    """
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

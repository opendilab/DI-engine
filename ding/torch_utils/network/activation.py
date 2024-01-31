import math
from collections.abc import Callable

import torch
import torch.nn as nn


class Lambda(nn.Module):
    """
    Overview:
        A custom lambda module for constructing custom layers.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, f: Callable):
        """
        Overview:
            Initialize the lambda module with a given function.
        Arguments:
            - f (:obj:`Callable`): a python function
        """
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute the function of the input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        """
        return self.f(x)


class GLU(nn.Module):
    """
    Overview:
        Gating Linear Unit (GLU), a specific type of activation function, which is first proposed in
        [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf).
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, input_dim: int, output_dim: int, context_dim: int, input_type: str = 'fc') -> None:
        """
        Overview:
            Initialize the GLU module.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input tensor.
            - output_dim (:obj:`int`): The dimension of the output tensor.
            - context_dim (:obj:`int`): The dimension of the context tensor.
            - input_type (:obj:`str`): The type of input, now supports ['fc', 'conv2d']
    """
        super(GLU, self).__init__()
        assert (input_type in ['fc', 'conv2d'])
        if input_type == 'fc':
            self.layer1 = nn.Linear(context_dim, input_dim)
            self.layer2 = nn.Linear(input_dim, output_dim)
        elif input_type == 'conv2d':
            self.layer1 = nn.Conv2d(context_dim, input_dim, 1, 1, 0)
            self.layer2 = nn.Conv2d(input_dim, output_dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute the GLU transformation of the input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
            - context (:obj:`torch.Tensor`): The context tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor after GLU transformation.
        """
        gate = self.layer1(context)
        gate = torch.sigmoid(gate)
        x = gate * x
        x = self.layer2(x)
        return x


class Swish(nn.Module):
    """
    Overview:
        Swish activation function, which is a smooth, non-monotonic activation function. For more details, please refer
        to [Searching for Activation Functions](https://arxiv.org/pdf/1710.05941.pdf).
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self):
        """
        Overview:
            Initialize the Swish module.
        """
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute the Swish transformation of the input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor after Swish transformation.
        """
        return x * torch.sigmoid(x)


class GELU(nn.Module):
    """
    Overview:
        Gaussian Error Linear Units (GELU) activation function, which is widely used in NLP models like GPT, BERT.
        For more details, please refer to the original paper: https://arxiv.org/pdf/1606.08415.pdf.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self):
        """
        Overview:
            Initialize the GELU module.
        """
        super(GELU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute the GELU transformation of the input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor after GELU transformation.
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def build_activation(activation: str, inplace: bool = None) -> nn.Module:
    """
    Overview:
        Build and return the activation module according to the given type.
    Arguments:
        - activation (:obj:`str`): The type of activation module, now supports \
            ['relu', 'glu', 'prelu', 'swish', 'gelu', 'tanh', 'sigmoid', 'softplus', 'elu', 'square', 'identity'].
        - inplace (Optional[:obj:`bool`): Execute the operation in-place in activation, defaults to None.
    Returns:
        - act_func (:obj:`nn.module`): The corresponding activation module.
    """
    if inplace is not None:
        assert activation == 'relu', 'inplace argument is not compatible with {}'.format(activation)
    else:
        inplace = False
    act_func = {
        'relu': nn.ReLU(inplace=inplace),
        'glu': GLU,
        'prelu': nn.PReLU(),
        'swish': Swish(),
        'gelu': GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "elu": nn.ELU(),
        "silu": torch.nn.SiLU(inplace=inplace),
        "square": Lambda(lambda x: x ** 2),
        "identity": Lambda(lambda x: x),
    }
    if activation.lower() in act_func.keys():
        return act_func[activation]
    else:
        raise KeyError("invalid key for activation: {}".format(activation))

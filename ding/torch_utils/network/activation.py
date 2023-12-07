import math

import torch
import torch.nn as nn


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class GLU(nn.Module):
    """
    Overview:
        Gating Linear Unit.
    Interfaces:
        ``forward``.

    .. tip::
        This module also supports 2D convolution, in which case, the input and context must have the same shape.
    """

    def __init__(self, input_dim: int, output_dim: int, context_dim: int, input_type: str = 'fc') -> None:
        """
        Overview:
            Init GLU
        Arguments:
            - input_dim (:obj:`int`): the input dimension
            - output_dim (:obj:`int`): the output dimension
            - context_dim (:obj:`int`): the context dimension
            - input_type (:obj:`str`): the type of input, now support ['fc', 'conv2d']
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
        r"""
        Overview:
            Return GLU computed tensor
        Arguments:
            - x (:obj:`torch.Tensor`) : the input tensor
            - context (:obj:`torch.Tensor`) : the context tensor
        Returns:
            - x (:obj:`torch.Tensor`): the computed tensor
        """
        gate = self.layer1(context)
        gate = torch.sigmoid(gate)
        x = gate * x
        x = self.layer2(x)
        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


class GELU(nn.Module):
    r"""
    Overview:
        Gaussian Error Linear Units (GELU) activation function, which is widely used in NLP models like GPT, BERT.
        The original paper can be viewed in: <link https://arxiv.org/pdf/1606.08415.pdf link>
    Interfaces:
        forward
    """

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SiLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def build_activation(activation: str, inplace: bool = None) -> nn.Module:
    """
    Overview:
        Return the activation module according to the given type.
    Arguments:
        - activation (:obj:`str`): The type of activation module, now supports \
            ['relu', 'glu', 'prelu', 'swish', 'gelu', 'tanh', 'sigmoid', 'softplus', 'elu', 'square', 'identity'].
        - inplace (:obj:`bool`): Execute the operation in-place in activation, defaults to ``None``.
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
        "silu": SiLU(),
        "square": Lambda(lambda x: x ** 2),
        "identity": Lambda(lambda x: x),
    }
    if activation.lower() in act_func.keys():
        return act_func[activation]
    else:
        raise KeyError("invalid key for activation: {}".format(activation))

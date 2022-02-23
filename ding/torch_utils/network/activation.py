import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    r"""
    Overview:
        Gating Linear Unit.
        This class does a thing like this:

        .. code:: python

            # Inputs: input, context, output_size
            # The gate value is a learnt function of the input.
            gate = sigmoid(linear(input.size)(context))
            # Gate the input and return an output of desired size.
            gated_input = gate * input
            output = linear(output_size)(gated_input)
            return output
    Interfaces:
        forward

    .. tip::

        This module also supports 2D convolution, in which case, the input and context must have the same shape.
    """

    def __init__(self, input_dim: int, output_dim: int, context_dim: int, input_type: str = 'fc') -> None:
        r"""
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

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def build_activation(activation: str, inplace: bool = None) -> nn.Module:
    r"""
    Overview:
        Return the activation module according to the given type.
    Arguments:
        - actvation (:obj:`str`): the type of activation module, now supports ['relu', 'glu', 'prelu']
        - inplace (:obj:`bool`): can optionally do the operation in-place in relu. Default ``None``
    Returns:
        - act_func (:obj:`nn.module`): the corresponding activation module
    """
    if inplace is not None:
        assert activation == 'relu', 'inplace argument is not compatible with {}'.format(activation)
    else:
        inplace = False
    act_func = {'relu': nn.ReLU(inplace=inplace), 'glu': GLU, 'prelu': nn.PReLU(), 'swish': Swish()}
    if activation in act_func.keys():
        return act_func[activation]
    else:
        raise KeyError("invalid key for activation: {}".format(activation))

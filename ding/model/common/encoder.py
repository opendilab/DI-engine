from typing import Optional
import torch
import torch.nn as nn

from ding.torch_utils import ResFCBlock, ResBlock, Flatten
from ding.utils import SequenceType


class ConvEncoder(nn.Module):
    """
    Overview:
        The ``Convolution Encoder`` used to encode raw 2-dim observations.
    Interfaces:
        ``__init__``, ``_get_flatten_size``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [32, 64, 64, 128],
            activation: Optional[nn.Module] = nn.ReLU(),
            kernel_size: SequenceType = [8, 4, 3],
            stride: SequenceType = [4, 2, 1],
            padding: Optional[SequenceType] = None,
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the ``Convolution Encoder`` according to the provided arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, plus one or more ``input size``.
            - hidden_size_list (:obj:`SequenceType`): Sequence of ``hidden_size`` of subsequent conv layers \
                and the final dense layer.
            - activation (:obj:`nn.Module`): Type of activation to use in the conv ``layers`` and ``ResBlock``. \
                Default is ``nn.ReLU()``.
            - kernel_size (:obj:`SequenceType`): Sequence of ``kernel_size`` of subsequent conv layers.
            - stride (:obj:`SequenceType`): Sequence of ``stride`` of subsequent conv layers.
            - padding (:obj:`SequenceType`): Padding added to all four sides of the input for each conv layer. \
                See ``nn.Conv2d`` for more details. Default is ``None``.
            - norm_type (:obj:`str`): Type of normalization to use. See ``ding.torch_utils.network.ResBlock`` \
                for more details. Default is ``None``.
        """
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list
        if padding is None:
            padding = [0 for _ in range(len(kernel_size))]

        layers = []
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i], padding[i]))
            layers.append(self.act)
            input_size = hidden_size_list[i]
        assert len(set(hidden_size_list[3:-1])) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(3, len(self.hidden_size_list) - 1):
            layers.append(ResBlock(self.hidden_size_list[i], activation=self.act, norm_type=norm_type))
        layers.append(Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.mid = nn.Linear(flatten_size, hidden_size_list[-1])

    def _get_flatten_size(self) -> int:
        """
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Returns:
            - outputs (:obj:`torch.Tensor`): Size ``int`` Tensor representing the number of ``in-features``.
        Shapes:
            - outputs: :math:`(1,)`.
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output embedding tensor of the env observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Shapes:
            - outputs: :math:`(B, N)`, where ``N = hidden_size_list[-1]``.
        """
        x = self.main(x)
        x = self.mid(x)
        return x


class FCEncoder(nn.Module):
    """
    Overview:
        The ``FCEncoder`` used in models to encode raw 1-dim observations.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: int,
            hidden_size_list: SequenceType,
            res_block: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the FC Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape.
            - hidden_size_list (:obj:`SequenceType`): Sequence of ``hidden_size`` of subsequent FC layers.
            - res_block (:obj:`bool`): Whether use ``res_block``. Default is ``False``.
            - activation (:obj:`nn.Module`): Type of activation to use in ``ResFCBlock``. Default is ``nn.ReLU()``.
            - norm_type (:obj:`str`): Type of normalization to use. See ``ding.torch_utils.network.ResFCBlock`` \
                for more details. Default is ``None``.
        """
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])

        if res_block:
            assert len(set(hidden_size_list)) == 1, "Please indicate the same hidden size for res block parts"
            if len(hidden_size_list) == 1:
                self.main = ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type)
            else:
                layers = []
                for i in range(len(hidden_size_list)):
                    layers.append(ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type))
                self.main = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(len(hidden_size_list) - 1):
                layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
                layers.append(self.act)
            self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output embedding tensor of the env observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Shapes:
            - outputs: :math:`(B, N)`, where ``N = hidden_size_list[-1]``.
        """
        x = self.act(self.init(x))
        x = self.main(x)
        return x


class StructEncoder(nn.Module):
    # TODO(nyz)
    pass

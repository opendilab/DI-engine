from typing import Optional
import torch
import torch.nn as nn

from ding.torch_utils import ResFCBlock, ResBlock, Flatten
from ding.utils import SequenceType


class ConvEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [32, 64, 64, 128],
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the Convolution Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, some ``output size``
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - activation (:obj:`nn.Module`):
                The type of activation to use in the conv ``layers`` and ``ResBlock``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResBlock`` for more details
        """
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list

        layers = []
        kernel_size = [8, 4, 3]
        stride = [4, 2, 1]
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i]))
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
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Arguments:
            - x (:obj:`torch.Tensor`): Encoded Tensor after ``self.main``
        Returns:
            - outputs (:obj:`torch.Tensor`): Size int, also number of in-feature
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.main(x)
        x = self.mid(x)
        return x


class FCEncoder(nn.Module):
    r"""
    Overview:
        The ``FCEncoder`` used in models. Used to encoder raw 1-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: int,
            hidden_size_list: SequenceType,
            res_block: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the FC Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - res_block (:obj:`bool`): Whether use ``res_block``.
            - activation (:obj:`nn.Module`):
                The type of activation to use in the ``ResFCBlock``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResFCBlock`` for more details
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
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.act(self.init(x))
        x = self.main(x)
        return x


class StructEncoder(nn.Module):
    # TODO(nyz)
    pass

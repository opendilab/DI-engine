"""Procedure cloning with BFS."""
import torch
import torch.nn as nn
from typing import Union, Optional

from ding.torch_utils import ResBlock, Flatten
from ding.utils import MODEL_REGISTRY, SequenceType


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
        x = self.main(x)
        return x


@MODEL_REGISTRY.register('pc')
class PC(nn.Module):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        encoder_hidden_size_list: SequenceType = [128, 128, 256, 256],
        augment=False
    ):
        super().__init__()

        self._augment = augment
        num_layers = len(encoder_hidden_size_list)

        kernel_sizes = (3, ) * (num_layers + 1)
        stride_sizes = (1, ) * (num_layers + 1)
        padding_sizes = (1, ) * (num_layers + 1)
        encoder_hidden_size_list.append(action_shape + 1)

        self._encoder = ConvEncoder(
            obs_shape=obs_shape,
            hidden_size_list=encoder_hidden_size_list,
            kernel_size=kernel_sizes,
            stride=stride_sizes,
            padding=padding_sizes,
        )

        # if self._augment:
        #     self._augment_layers = nn.Sequential([
        #         tf.keras.layers.RandomCrop(maze_size, maze_size),
        #         tf.keras.layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1),
        #                                           fill_mode='constant'),
        #         tf.keras.layers.RandomZoom((-0.1, 0.1), (-0.1, 0.1),
        #                                    fill_mode='constant'),
        #     ])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self._encoder(x)
        return {'logit': x.permute(0, 2, 3, 1)}

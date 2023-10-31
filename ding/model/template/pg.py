from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn
from easydict import EasyDict

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, \
        MultiHead, RegressionHead, ReparameterizationHead, independent_normal_dist


@MODEL_REGISTRY.register('pg')
class PG(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to Policy Gradient(PG) \
        (https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf). \
        The PG model is composed of two parts: encoder and head. Encoders are used to extract the feature \
        from various observation. Heads are used to predict corresponding action logit.
    Interface:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            action_space: str = 'discrete',
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Initialize the PG model according to corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - action_space (:obj:`str`): The type of different action spaces, including ['discrete', 'continuous'], \
                then will instantiate corresponding head, including ``DiscreteHead`` and ``ReparameterizationHead``.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``head`` network, defaults \
                to None, it must match the last element of ``encoder_hidden_size_list``.
            - head_layer_num (:obj:`int`): The num of layers used in the ``head`` network to compute action.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN']
        Examples:
            >>> model = PG((4, 84, 84), 5)
            >>> inputs = torch.randn(8, 4, 84, 84)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == (8, 5)
            >>> assert outputs['dist'].sample().shape == (8, )
        """
        super(PG, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own BC".format(obs_shape)
            )
        self.action_space = action_space
        # Head
        if self.action_space == 'discrete':
            self.head = DiscreteHead(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )
        elif self.action_space == 'continuous':
            self.head = ReparameterizationHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
                sigma_type='independent'
            )
        else:
            raise KeyError("not support action space: {}".format(self.action_space))

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            PG forward computation graph, input observation tensor to predict policy distribution.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`torch.distributions`): The output policy distribution. If action space is \
            discrete, the output is Categorical distribution; if action space is continuous, the output is Normal \
            distribution.
        """
        x = self.encoder(x)
        x = self.head(x)
        if self.action_space == 'discrete':
            x['dist'] = torch.distributions.Categorical(logits=x['logit'])
        elif self.action_space == 'continuous':
            x = {'logit': {'mu': x['mu'], 'sigma': x['sigma']}}
            x['dist'] = independent_normal_dist(x['logit'])
        return x

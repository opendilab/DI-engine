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
        x = self.encoder(x)
        x = self.head(x)
        if self.action_space == 'discrete':
            x['dist'] = torch.distributions.Categorical(logits=x['logit'])
        elif self.action_space == 'continuous':
            x = {'logit': {'mu': x['mu'], 'sigma': x['sigma']}}
            x['dist'] = independent_normal_dist(x['logit'])
        return x

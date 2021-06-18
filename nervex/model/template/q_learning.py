from typing import Union, Optional, Dict
import torch
import torch.nn as nn

from nervex.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, ClassificationHead, DuelingHead, MultiDiscreteHead


@MODEL_REGISTRY.register('dqn')
class DQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: int = 64,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(DQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = ClassificationHead
        multi_discrete = not isinstance(action_shape, int)
        if multi_discrete:
            self.head = MultiDiscreteHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


class C51DQN(nn.Module):
    pass


class QRDQN(nn.Module):
    pass


class IQN(nn.Module):
    pass


class RainbowDQN(nn.Module):
    pass


class DRQN(nn.Module):
    pass


class GeneralQNetwork(nn.Module):
    pass

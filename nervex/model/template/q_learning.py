from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from nervex.torch_utils import get_lstm
from nervex.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, ClassificationHead, DuelingHead, MultiDiscreteHead, RainbowHead, QRDQNHead


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


@MODEL_REGISTRY.register('qrdqn')
class QRDQN(nn.Module):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        dueling: bool = False,
        head_hidden_size: int = 64,
        head_layer_num: int = 1,
        num_quantiles: int = 32,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        noise: Optional[bool] = False,
    ) -> None:
        super(QRDQN, self).__init__()
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
        multi_discrete = not isinstance(action_shape, int)
        if multi_discrete:
            self.head = MultiDiscreteHead(
                QRDQNHead,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                num_quantiles=num_quantiles,
                activation=activation,
                norm_type=norm_type,
                noise=noise,
            )
        else:
            self.head = QRDQNHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``,  ``q``, ``tau``
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


class IQN(nn.Module):
    pass


@MODEL_REGISTRY.register('rainbowdqn')
class RainbowDQN(nn.Module):
    """
    .. note::
        RainbowDQN contains dueling architecture by default
    """

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        head_hidden_size: int = 64,
        head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        v_min: Optional[float] = -10,
        v_max: Optional[float] = -10,
        n_atom: Optional[int] = 51,
    ) -> None:
        super(RainbowDQN, self).__init__()
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
        multi_discrete = not isinstance(action_shape, int)
        if multi_discrete:
            self.head = MultiDiscreteHead(
                RainbowHead,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type,
                n_atom=n_atom,
                v_min=v_min,
                v_max=v_max,
            )
        else:
            self.head = RainbowHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
                n_atom=n_atom,
                v_min=v_min,
                v_max=v_max,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``, ``distribution``
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


class SQN(nn.Module):
    pass


def parallel_wrapper(forward_fn: Callable) -> Callable:
    r"""
    Overview:
        Process timestep T and batch_size B at the same time, in other words, treat different timestep data as
        different trajectories in a batch. Used in ``DiscreteNet``'s ``fast_timestep_forward``.
    Arguments:
        - forward_fn (:obj:`Callable`): normal nn.Module's forward function
    Returns:
        - wrapper (:obj:`Callable`): wrapped function
    """

    def wrapper(x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        T, B = x.shape[:2]

        def reshape(d):
            if isinstance(d, list):
                d = [reshape(t) for t in d]
            elif isinstance(d, dict):
                d = {k: reshape(v) for k, v in d.items()}
            else:
                d = d.reshape(T, B, *d.shape[1:])
            return d

        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        x = reshape(x)
        return x

    return wrapper


@MODEL_REGISTRY.register('drqn')
class DRQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: int = 64,
            head_layer_num: int = 1,
            lstm_type: Optional[str] = 'normal',
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(DRQN, self).__init__()
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
        # LSTM Type
        self.rnn = get_lstm(lstm_type, input_size=head_hidden_size, hidden_size=head_hidden_size)
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

    def forward(self, inputs: Dict, inference: bool = False) -> Dict:
        """
        ArgumentsKeys:
            - necessary: ``obs``, ``prev_state``
        ReturnsKeys:
            - necessary: ``logit``, ``next_state``
        """
        x, prev_state = inputs['obs'], inputs['prev_state']
        if inference:
            x = self.encoder(x)
            x = x.unsqueeze(0)
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)
            x = self.head(x)
            x['next_state'] = next_state
            return x
        else:
            assert len(x.shape) in [3, 5], x.shape
            x = parallel_wrapper(self.encoder)(x)
            lstm_embedding = []
            for t in range(x.shape[0]):  # T timesteps
                output, prev_state = self.rnn(x[t:t + 1], prev_state)
                lstm_embedding.append(output)
            x = torch.cat(lstm_embedding, 0)
            x = parallel_wrapper(self.head)(x)
            x['next_state'] = prev_state
            return x


class GeneralQNetwork(nn.Module):
    pass

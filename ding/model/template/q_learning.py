from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead, RainbowHead, \
    QuantileHead, QRDQNHead, DistributionHead


@MODEL_REGISTRY.register('dqn')
class DQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the DQN (encoder + head) Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - dueling (:obj:`dueling`): Whether choose ``DuelingHead`` or ``DiscreteHead(default)``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details.
        """
        super(DQN, self).__init__()
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
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
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
        r"""
        Overview:
            DQN forward computation graph, input observation tensor to predict q_value.
        Arguments:
            - x (:obj:`torch.Tensor`): Observation inputs
        Returns:
            - outputs (:obj:`Dict`): DQN forward outputs, such as q_value.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each action dimension.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``
        Examples:
            >>> model = DQN(32, 6)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 32)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 6])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('c51dqn')
class C51DQN(nn.Module):

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
        v_max: Optional[float] = 10,
        n_atom: Optional[int] = 51,
    ) -> None:
        r"""
        Overview:
            Init the C51 Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
            - n_atom (:obj:`Optional[int]`): Number of atoms in the prediction distribution.
        """
        super(C51DQN, self).__init__()
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
                "not support obs_shape for pre-defined encoder: {}, please customize your own C51DQN".format(obs_shape)
            )
        # Head Type
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                DistributionHead,
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
            self.head = DistributionHead(
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
        r"""
        Overview:
            Use observation tensor to predict C51DQN's output.
            Parameter updates with C51DQN's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor w/ ``(B, N=head_hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head. Return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - distribution (:obj:`torch.Tensor`): Distribution tensor of size ``(B, N, n_atom)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape.
            - distribution(:obj:`torch.FloatTensor`): :math:`(B, M, P)`, where P is n_atom.

        Examples:
            >>> model = C51DQN(128, 64)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 128)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> # default head_hidden_size: int = 64,
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default n_atom: int = 51
            >>> assert outputs['distribution'].shape == torch.Size([4, 64, 51])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('qrdqn')
class QRDQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            num_quantiles: int = 32,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the QRDQN Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - num_quantiles (:obj:`int`): Number of quantiles in the prediction distribution.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(QRDQN, self).__init__()
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
                "not support obs_shape for pre-defined encoder: {}, please customize your own QRDQN".format(obs_shape)
            )
        # Head Type
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                QRDQNHead,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                num_quantiles=num_quantiles,
                activation=activation,
                norm_type=norm_type,
            )
        else:
            self.head = QRDQNHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                num_quantiles=num_quantiles,
                activation=activation,
                norm_type=norm_type,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict QRDQN's output.
            Parameter updates with QRDQN's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor with ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head. Return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - q (:obj:`torch.Tensor`): Q valye tensor tensor of size ``(B, N, num_quantiles)``
            - tau (:obj:`torch.Tensor`): tau tensor of size ``(B, N, 1)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape.
            - tau (:obj:`torch.Tensor`):  :math:`(B, M, 1)`

        Examples:
            >>> model = QRDQN(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles : int = 32
            >>> assert outputs['q'].shape == torch.Size([4, 64, 32])
            >>> assert outputs['tau'].shape == torch.Size([4, 32, 1])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('iqn')
class IQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            num_quantiles: int = 32,
            quantile_embedding_size: int = 128,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the IQN Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape.
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - num_quantiles (:obj:`int`): Number of quantiles in the prediction distribution.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details.
        """
        super(IQN, self).__init__()
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
                "not support obs_shape for pre-defined encoder: {}, please customize your own IQN".format(obs_shape)
            )
        # Head Type
        head_cls = QuantileHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                num_quantiles=num_quantiles,
                quantile_embedding_size=quantile_embedding_size,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
                num_quantiles=num_quantiles,
                quantile_embedding_size=quantile_embedding_size,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict IQN's output.
            Parameter updates with IQN's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor with ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head. Return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - q (:obj:`torch.Tensor`): Q valye tensor tensor of size ``(num_quantiles, N, B)``
            - quantiles (:obj:`torch.Tensor`): quantiles tensor of size ``(quantile_embedding_size, 1)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape
            - quantiles (:obj:`torch.Tensor`):  :math:`(P, 1)`, where P is quantile_embedding_size.
        Examples:
            >>> model = IQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles: int = 32
            >>> assert outputs['q'].shape == torch.Size([32, 4, 64]
            >>> # default quantile_embedding_size: int = 128
            >>> assert outputs['quantiles'].shape == torch.Size([128, 1])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('rainbowdqn')
class RainbowDQN(nn.Module):
    """
    Overview:
        RainbowDQN network (C51 + Dueling + Noisy Block)

    .. note::
        RainbowDQN contains dueling architecture by default
    """

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        head_hidden_size: Optional[int] = None,
        head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        v_min: Optional[float] = -10,
        v_max: Optional[float] = 10,
        n_atom: Optional[int] = 51,
    ) -> None:
        """
        Overview:
            Init the Rainbow Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape.
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` the after \
                ``layer_fn``, if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization to use, see ``ding.torch_utils.fc_block`` \
                for more details`
            - n_atom (:obj:`Optional[int]`): Number of atoms in the prediction distribution.
        """
        super(RainbowDQN, self).__init__()
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
                "not support obs_shape for pre-defined encoder: {}, please customize your own RainbowDQN".
                format(obs_shape)
            )
        # Head Type
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
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
        r"""
        Overview:
            Use observation tensor to predict Rainbow output.
            Parameter updates with Rainbow's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor with ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``RainbowHead`` setups and return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - distribution (:obj:`torch.Tensor`): Distribution tensor of size ``(B, N, n_atom)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape.
            - distribution(:obj:`torch.FloatTensor`): :math:`(B, M, P)`, where P is n_atom.

        Examples:
            >>> model = RainbowDQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default n_atom: int =51
            >>> assert outputs['distribution'].shape == torch.Size([4, 64, 51])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


def parallel_wrapper(forward_fn: Callable) -> Callable:
    r"""
    Overview:
        Process timestep T and batch_size B at the same time, in other words, treat different timestep data as
        different trajectories in a batch.
    Arguments:
        - forward_fn (:obj:`Callable`): Normal ``nn.Module`` 's forward function.
    Returns:
        - wrapper (:obj:`Callable`): Wrapped function.
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
    """
    Overview:
        DQN + RNN = DRQN
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            lstm_type: Optional[str] = 'normal',
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the DRQN Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - lstm_type (:obj:`Optional[str]`): Version of lstm cell, now support ``['normal', 'pytorch']``
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(DRQN, self).__init__()
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
                "not support obs_shape for pre-defined encoder: {}, please customize your own DRQN".format(obs_shape)
            )
        # LSTM Type
        self.rnn = get_lstm(lstm_type, input_size=head_hidden_size, hidden_size=head_hidden_size)
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
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
        r"""
        Overview:
            Use observation tensor to predict DRQN output.
            Parameter updates with DRQN's MLPs forward setup.
        Arguments:
            - inputs (:obj:`Dict`):

       ArgumentsKeys:
            - obs (:obj:`torch.Tensor`): Encoded observation
            - prev_state (:obj:`list`): Previous state's tensor of size ``(B, N)``

        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``DRQN`` setups and return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``obs``.
            - next_state (:obj:`list`): Next state's tensor of size ``(B, N)``
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N=obs_space)`, where B is batch size.
            - prev_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - next_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`

        Examples:
            >>> # Init input's Keys:
            >>> prev_state = [[torch.randn(1, 1, 64) for __ in range(2)] for _ in range(4)] # B=4
            >>> obs = torch.randn(4,64)
            >>> model = DRQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=True)
            >>> # Check outputs's Keys
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == (4, 64)
            >>> assert len(outputs['next_state']) == 4
            >>> assert all([len(t) == 2 for t in outputs['next_state']])
            >>> assert all([t[0].shape == (1, 1, 64) for t in outputs['next_state']])
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

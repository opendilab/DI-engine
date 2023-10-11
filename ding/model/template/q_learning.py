from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead, RainbowHead, \
    QuantileHead, FQFHead, QRDQNHead, DistributionHead, BranchingHead
from ding.torch_utils.network.gtrxl import GTrXL


@MODEL_REGISTRY.register('dqn')
class DQN(nn.Module):
    """
    Overview:
        The neural nework structure and computation graph of Deep Q Network (DQN) algorithm, which is the most classic \
        value-based RL algorithm for discrete action. The DQN is composed of two parts: ``encoder`` and ``head``. \
        The ``encoder`` is used to extract the feature from various observation, and the ``head`` is used to compute \
        the Q value of each action dimension.
    Interfaces:
        ``__init__``, ``forward``.

    .. note::
        Current ``DQN`` supports two types of encoder: ``FCEncoder`` and ``ConvEncoder``, two types of head: \
        ``DiscreteHead`` and ``DuelingHead``. You can customize your own encoder or head by inheriting this class.
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            dropout: Optional[float] = None
    ) -> None:
        """
        Overview:
            initialize the DQN (encoder + head) Model according to corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - dueling (:obj:`Optional[bool]`): Whether choose ``DuelingHead`` or ``DiscreteHead (default)``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network, defaults to None, \
                then it will be set to the last element of ``encoder_hidden_size_list``.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN']
            - dropout (:obj:`Optional[float]`): The dropout rate of the dropout layer. \
                if ``None`` then default disable dropout layer.
        """
        super(DQN, self).__init__()
        # Squeeze data from tuple, list or dict to single object. For example, from (4, ) to 4
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type, dropout=dropout
            )
        # Conv Encoder
        elif len(obs_shape) == 3:
            assert dropout is None, "dropout is not supported in ConvEncoder"
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
                norm_type=norm_type,
                dropout=dropout
            )
        else:
            self.head = head_cls(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
                dropout=dropout
            )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            DQN forward computation graph, input observation tensor to predict q_value.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict`): The output of DQN's forward, including q_value.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each possible action dimension.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.Tensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``
        Examples:
            >>> model = DQN(32, 6)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 32)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 6])

        .. note::
            For consistency and compatibility, we name all the outputs of the network which are related to action \
            selections as ``logit``.
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('bdq')
class BDQ(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            num_branches: int = 0,
            action_bins_per_branch: int = 2,
            layer_num: int = 3,
            a_layer_num: Optional[int] = None,
            v_layer_num: Optional[int] = None,
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            head_hidden_size: Optional[int] = None,
            norm_type: Optional[nn.Module] = None,
            activation: Optional[nn.Module] = nn.ReLU(),
    ) -> None:
        """
        Overview:
            Init the BDQ (encoder + head) Model according to input arguments. \
                referenced paper Action Branching Architectures for Deep Reinforcement Learning \
                <https://arxiv.org/pdf/1711.08946>
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - num_branches (:obj:`int`): The number of branches, which is equivalent to the action dimension, \
                such as 6 in mujoco's halfcheetah environment.
            - action_bins_per_branch (:obj:`int`): The number of actions in each dimension.
            - layer_num (:obj:`int`): The number of layers used in the network to compute Advantage and Value output.
            - a_layer_num (:obj:`int`): The number of layers used in the network to compute Advantage output.
            - v_layer_num (:obj:`int`): The number of layers used in the network to compute Value output.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``
        """
        super(BDQ, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, num_branches = squeeze(obs_shape), squeeze(num_branches)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]

        # backbone
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

        self.num_branches = num_branches
        self.action_bins_per_branch = action_bins_per_branch

        # head
        self.head = BranchingHead(
            head_hidden_size,
            num_branches=self.num_branches,
            action_bins_per_branch=self.action_bins_per_branch,
            layer_num=layer_num,
            a_layer_num=a_layer_num,
            v_layer_num=v_layer_num,
            activation=activation,
            norm_type=norm_type
        )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            BDQ forward computation graph, input observation tensor to predict q_value.
        Arguments:
            - x (:obj:`torch.Tensor`): Observation inputs
        Returns:
            - outputs (:obj:`Dict`): BDQ forward outputs, such as q_value.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each action dimension.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is
                ``num_branches * action_bins_per_branch``
        Examples:
            >>> model = BDQ(8, 5, 2)  # arguments: 'obs_shape', 'num_branches' and 'action_bins_per_branch'.
            >>> inputs = torch.randn(4, 8)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 5, 2])
        """
        x = self.encoder(x) / (self.num_branches + 1)  # corresponds to the "Gradient Rescaling" in the paper
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('c51dqn')
class C51DQN(nn.Module):
    """
    Overview:
        The neural network structure and computation graph of C51DQN, which combines distributional RL and DQN. \
        You can refer to https://arxiv.org/pdf/1707.06887.pdf for more details. The C51DQN is composed of \
        ``encoder`` and ``head``. ``encoder`` is used to extract the feature of observation, and ``head`` is \
        used to compute the distribution of Q-value.
    Interfaces:
        ``__init__``, ``forward``

    .. note::
        Current C51DQN supports two types of encoder: ``FCEncoder`` and ``ConvEncoder``.
    """

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        head_hidden_size: int = None,
        head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        v_min: Optional[float] = -10,
        v_max: Optional[float] = 10,
        n_atom: Optional[int] = 51,
    ) -> None:
        """
        Overview:
            initialize the C51 Model according to corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network, defaults to None, \
                then it will be set to the last element of ``encoder_hidden_size_list``.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN']
            - v_min (:obj:`Optional[float]`): The minimum value of the support of the distribution, which is related \
                to the value (discounted sum of reward) scale of the specific environment. Defaults to -10.
            - v_max (:obj:`Optional[float]`): The maximum value of the support of the distribution, which is related \
                to the value (discounted sum of reward) scale of the specific environment. Defaults to 10.
            - n_atom (:obj:`Optional[int]`): The number of atoms in the prediction distribution, 51 is the default \
                value in the paper, you can also try other values such as 301.
        """
        super(C51DQN, self).__init__()
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
        """
        Returns:
            - outputs (:obj:`Dict`): The output of DQN's forward, including q_value.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each possible action dimension.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.Tensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``
        Overview:
            C51DQN forward computation graph, input observation tensor to predict q_value and its distribution.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict`): The output of DQN's forward, including q_value, and distribution.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each possible action dimension.
            - distribution (:obj:`torch.Tensor`): Q-Value discretized distribution, i.e., probability of each \
                uniformly spaced atom Q-value, such as dividing [-10, 10] into 51 uniform spaces.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.Tensor`): :math:`(B, M)`, where M is action_shape.
            - distribution(:obj:`torch.Tensor`): :math:`(B, M, P)`, where P is n_atom.

        Examples:
            >>> model = C51DQN(128, 64)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 128)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> # default head_hidden_size: int = 64,
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default n_atom: int = 51
            >>> assert outputs['distribution'].shape == torch.Size([4, 64, 51])

        .. note::
            For consistency and compatibility, we name all the outputs of the network which are related to action \
            selections as ``logit``.

        .. note::
            For convenience, we recommend that the number of atoms should be odd, so that the middle atom is exactly \
            the value of the Q-value.
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


@MODEL_REGISTRY.register('fqf')
class FQF(nn.Module):

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
            Init the FQF Model according to input arguments.
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
        super(FQF, self).__init__()
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
                "not support obs_shape for pre-defined encoder: {}, please customize your own FQF".format(obs_shape)
            )
        # Head Type
        head_cls = FQFHead
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
            Use encoded embedding tensor to predict FQF's output.
            Parameter updates with FQF's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor with ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``logit`` (:obj:`torch.Tensor`), \
                    ``q`` (:obj:`torch.Tensor`), ``quantiles`` (:obj:`torch.Tensor`), \
                    ``quantiles_hats`` (:obj:`torch.Tensor`), \
                    ``q_tau_i`` (:obj:`torch.Tensor`), ``entropies`` (:obj:`torch.Tensor`).
        Shapes:
            - x: :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit: :math:`(B, M)`, where M is action_shape.
            - q: :math:`(B, num_quantiles, M)`.
            - quantiles: :math:`(B, num_quantiles + 1)`.
            - quantiles_hats: :math:`(B, num_quantiles)`.
            - q_tau_i: :math:`(B, num_quantiles - 1, M)`.
            - entropies: :math:`(B, 1)`.
        Examples:
            >>> model = FQF(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles: int = 32
            >>> assert outputs['q'].shape == torch.Size([4, 32, 64])
            >>> assert outputs['quantiles'].shape == torch.Size([4, 33])
            >>> assert outputs['quantiles_hats'].shape == torch.Size([4, 32])
            >>> assert outputs['q_tau_i'].shape == torch.Size([4, 31, 64])
            >>> assert outputs['quantiles'].shape == torch.Size([4, 1])
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
        RainbowDQN contains dueling architecture by default.
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

        # NOTE(rjy): the initial input shape will be (T, B, N),
        #            means encoder or head should process B trajectorys, each trajectory has T timestep,
        #            but T and B dimension can be both treated as batch_size in encoder and head,
        #            i.e., independent and parallel processing,
        #            so here we need such fn to reshape for encoder or head
        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        x = reshape(x)
        return x

    return wrapper


@MODEL_REGISTRY.register('drqn')
class DRQN(nn.Module):
    """
    Overview:
        The neural network structure and computation graph of DRQN (DQN + RNN = DRQN) algorithm, which is the most \
        common DQN variant for sequential data and paratially observable environment. The DRQN is composed of three \
        parts: ``encoder``, ``head`` and ``rnn``. The ``encoder`` is used to extract the feature from various \
        observation, the ``rnn`` is used to process the sequential observation and other data, and the ``head`` is \
        used to compute the Q value of each action dimension.
    Interfaces:
        ``__init__``, ``forward``.

    .. note::
        Current ``DRQN`` supports two types of encoder: ``FCEncoder`` and ``ConvEncoder``, two types of head: \
        ``DiscreteHead`` and ``DuelingHead``, three types of rnn: ``normal (LSTM with LayerNorm)``, ``pytorch`` and \
        ``gru``. You can customize your own encoder, rnn or head by inheriting this class.
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
            norm_type: Optional[str] = None,
            res_link: bool = False
    ) -> None:
        """
        Overview:
            Initialize the DRQN Model according to the corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - dueling (:obj:`Optional[bool]`): Whether choose ``DuelingHead`` or ``DiscreteHead (default)``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network, defaults to None, \
                then it will be set to the last element of ``encoder_hidden_size_list``.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output.
            - lstm_type (:obj:`Optional[str]`): The type of RNN module, now support ['normal', 'pytorch', 'gru'].
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN']
            - res_link (:obj:`bool`): Whether to enable the residual link, which is the skip connnection between \
                single frame data and the sequential data, defaults to False.
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
        self.res_link = res_link
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

    def forward(self, inputs: Dict, inference: bool = False, saved_state_timesteps: Optional[list] = None) -> Dict:
        """
        Overview:
            DRQN forward computation graph, input observation tensor to predict q_value.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The dict of input data, including observation and previous rnn state.
            - inference: (:obj:'bool'): Whether to enable inference forward mode, if True, we unroll the one timestep \
                transition, otherwise, we unroll the eentire sequence transitions.
            - saved_state_timesteps: (:obj:'Optional[list]'): When inference is False, we unroll the sequence \
                transitions, then we would use this list to indicate how to save and return hidden state.
        ArgumentsKeys:
            - obs (:obj:`torch.Tensor`): The raw observation tensor.
            - prev_state (:obj:`list`): The previous rnn state tensor, whose structure depends on ``lstm_type``.
        Returns:
            - outputs (:obj:`Dict`): The output of DRQN's forward, including logit (q_value) and next state.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each possible action dimension.
            - next_state (:obj:`list`): The next rnn state tensor, whose structure depends on ``lstm_type``.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.Tensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``

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
        # for both inference and other cases, the network structure is encoder -> rnn network -> head
        # the difference is inference take the data with seq_len=1 (or T = 1)
        # NOTE(rjy): in most situations, set inference=True when evaluate and inference=False when training
        if inference:
            x = self.encoder(x)
            if self.res_link:
                a = x
            x = x.unsqueeze(0)  # for rnn input, put the seq_len of x as 1 instead of none.
            # prev_state: DataType: List[Tuple[torch.Tensor]]; Initially, it is a list of None
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)  # to delete the seq_len dim to match head network input
            if self.res_link:
                x = x + a
            x = self.head(x)
            x['next_state'] = next_state
            return x
        else:
            # In order to better explain why rnn needs saved_state and which states need to be stored,
            # let's take r2d2 as an example
            # in r2d2,
            # 1) data['burnin_nstep_obs'] = data['obs'][:bs + self._nstep]
            # 2) data['main_obs'] = data['obs'][bs:-self._nstep]
            # 3) data['target_obs'] = data['obs'][bs + self._nstep:]
            # NOTE(rjy): (T, B, N) or (T, B, C, H, W)
            assert len(x.shape) in [3, 5], x.shape
            x = parallel_wrapper(self.encoder)(x)  # (T, B, N)
            if self.res_link:
                a = x
            # NOTE(rjy) lstm_embedding stores all hidden_state
            lstm_embedding = []
            # TODO(nyz) how to deal with hidden_size key-value
            hidden_state_list = []
            if saved_state_timesteps is not None:
                saved_state = []
            for t in range(x.shape[0]):  # T timesteps
                # NOTE(rjy) use x[t:t+1] but not x[t] can keep original dimension
                output, prev_state = self.rnn(x[t:t + 1], prev_state)  # output: (1,B, head_hidden_size)
                if saved_state_timesteps is not None and t + 1 in saved_state_timesteps:
                    saved_state.append(prev_state)
                lstm_embedding.append(output)
                hidden_state = [p['h'] for p in prev_state]
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
                hidden_state_list.append(torch.cat(hidden_state, dim=1))
            x = torch.cat(lstm_embedding, 0)  # (T, B, head_hidden_size)
            if self.res_link:
                x = x + a
            x = parallel_wrapper(self.head)(x)  # (T, B, action_shape)
            # NOTE(rjy): x['next_state'] is the hidden state of the last timestep inputted to lstm
            # the last timestep state including the hidden state (h) and the cell state (c)
            # shape: {list: B{dict: 2{Tensor:(1, 1, head_hidden_size}}}
            x['next_state'] = prev_state
            # all hidden state h, this returns a tensor of the dim: seq_len*batch_size*head_hidden_size
            # This key is used in qtran, the algorithm requires to retain all h_{t} during training
            x['hidden_state'] = torch.cat(hidden_state_list, dim=0)
            if saved_state_timesteps is not None:
                # the selected saved hidden states, including the hidden state (h) and the cell state (c)
                # in r2d2, set 'saved_hidden_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]',
                # then saved_state will record the hidden_state for main_obs and target_obs to
                # initialize their lstm (h c)
                x['saved_state'] = saved_state
            return x


@MODEL_REGISTRY.register('gtrxldqn')
class GTrXLDQN(nn.Module):
    """
    Overview:
        The neural network structure and computation graph of Gated Transformer-XL DQN algorithm, which is the \
        enhanced version of DRQN, using Transformer-XL to improve long-term sequential modelling ability. The \
        GTrXL-DQN is composed of three parts: ``encoder``, ``head`` and ``core``. The ``encoder`` is used to extract \
        the feature from various observation, the ``core`` is used to process the sequential observation and other \
        data, and the ``head`` is used to compute the Q value of each action dimension.
    Interfaces:
        ``__init__``, ``forward``, ``reset_memory``, ``get_memory`` .
    """

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        head_layer_num: int = 1,
        att_head_dim: int = 16,
        hidden_size: int = 16,
        att_head_num: int = 2,
        att_mlp_num: int = 2,
        att_layer_num: int = 3,
        memory_len: int = 64,
        activation: Optional[nn.Module] = nn.ReLU(),
        head_norm_type: Optional[str] = None,
        dropout: float = 0.,
        gru_gating: bool = True,
        gru_bias: float = 2.,
        dueling: bool = True,
        encoder_hidden_size_list: SequenceType = [128, 128, 256],
        encoder_norm_type: Optional[str] = None,
    ) -> None:
        """
        Overview:
            Initialize the GTrXLDQN model accoding to corresponding input arguments.

        .. tip::
            You can refer to GTrXl class in ``ding.torch_utils.network.gtrxl`` for more details about the input \
            arguments.

        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Used by Transformer. Observation's space.
            - action_shape (:obj:Union[int, SequenceType]): Used by Head. Action's space.
            - head_layer_num (:obj:`int`): Used by Head. Number of layers.
            - att_head_dim (:obj:`int`): Used by Transformer.
            - hidden_size (:obj:`int`): Used by Transformer and Head.
            - att_head_num (:obj:`int`): Used by Transformer.
            - att_mlp_num (:obj:`int`): Used by Transformer.
            - att_layer_num (:obj:`int`): Used by Transformer.
            - memory_len (:obj:`int`): Used by Transformer.
            - activation (:obj:`Optional[nn.Module]`): Used by Transformer and Head. if ``None`` then default set to \
                ``nn.ReLU()``.
            - head_norm_type (:obj:`Optional[str]`): Used by Head. The type of normalization to use, see \
                ``ding.torch_utils.fc_block`` for more details`.
            - dropout (:obj:`bool`): Used by Transformer.
            - gru_gating (:obj:`bool`): Used by Transformer.
            - gru_bias (:obj:`float`): Used by Transformer.
            - dueling (:obj:`bool`): Used by Head. Make the head dueling.
            - encoder_hidden_size_list(:obj:`SequenceType`): Used by Encoder. The collection of ``hidden_size`` if \
                using a custom convolutional encoder.
            - encoder_norm_type (:obj:`Optional[str]`): Used by Encoder. The type of normalization to use, see \
             ``ding.torch_utils.fc_block`` for more details`.
        """
        super(GTrXLDQN, self).__init__()
        self.core = GTrXL(
            input_dim=obs_shape,
            head_dim=att_head_dim,
            embedding_dim=hidden_size,
            head_num=att_head_num,
            mlp_num=att_mlp_num,
            layer_num=att_layer_num,
            memory_len=memory_len,
            activation=activation,
            dropout_ratio=dropout,
            gru_gating=gru_gating,
            gru_bias=gru_bias,
        )

        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            raise NotImplementedError("not support obs_shape for pre-defined encoder: {}".format(obs_shape))
        # replace the embedding layer of Transformer with Conv Encoder
        elif len(obs_shape) == 3:
            assert encoder_hidden_size_list[-1] == hidden_size
            self.obs_encoder = ConvEncoder(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=encoder_norm_type
            )
            self.dropout = nn.Dropout(dropout)
            self.core.use_embedding_layer = False
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own GTrXL".format(obs_shape)
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
                hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=head_norm_type
            )
        else:
            self.head = head_cls(
                hidden_size, action_shape, head_layer_num, activation=activation, norm_type=head_norm_type
            )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            Let input tensor go through GTrXl and the Head sequentially.
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor of shape (seq_len, bs, obs_shape).
        Returns:
            - out (:obj:`Dict`): run ``GTrXL`` with ``DiscreteHead`` setups and return the result prediction dictionary.
        ReturnKeys:
            - logit (:obj:`torch.Tensor`): discrete Q-value output of each action dimension, shape is (B, action_space).
            - memory (:obj:`torch.Tensor`): memory tensor of size ``(bs x layer_num+1 x memory_len x embedding_dim)``.
            - transformer_out (:obj:`torch.Tensor`): output tensor of transformer with same size as input ``x``.
        Examples:
            >>> # Init input's Keys:
            >>> obs_dim, seq_len, bs, action_dim = 128, 64, 32, 4
            >>> obs = torch.rand(seq_len, bs, obs_dim)
            >>> model = GTrXLDQN(obs_dim, action_dim)
            >>> outputs = model(obs)
            >>> assert isinstance(outputs, dict)
        """
        if len(x.shape) == 5:
            # 3d obs: cur_seq, bs, ch, h, w
            x_ = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[-3:]))
            x_ = self.dropout(self.obs_encoder(x_))
            x = x_.reshape(x.shape[0], x.shape[1], -1)
        o1 = self.core(x)
        out = self.head(o1['logit'])
        # layer_num+1 x memory_len x bs embedding_dim -> bs x layer_num+1 x memory_len x embedding_dim
        out['memory'] = o1['memory'].permute((2, 0, 1, 3)).contiguous()
        out['transformer_out'] = o1['logit']  # output of gtrxl, out['logit'] is final output
        return out

    def reset_memory(self, batch_size: Optional[int] = None, state: Optional[torch.Tensor] = None) -> None:
        """
        Overview:
            Clear or reset the memory of GTrXL.
        Arguments:
            - batch_size (:obj:`Optional[int]`): The number of samples in a training batch.
            - state (:obj:`Optional[torch.Tensor]`): The input memory data, whose shape is \
                (layer_num, memory_len, bs, embedding_dim).
        """
        self.core.reset_memory(batch_size, state)

    def get_memory(self) -> Optional[torch.Tensor]:
        """
        Overview:
            Return the memory of GTrXL.
        Returns:
            - memory: (:obj:`Optional[torch.Tensor]`): output memory or None if memory has not been initialized, \
                whose shape is (layer_num, memory_len, bs, embedding_dim).
        """
        return self.core.get_memory()

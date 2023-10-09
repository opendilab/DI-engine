from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn

from ding.utils import MODEL_REGISTRY, SequenceType
from ding.torch_utils.network.transformer import Attention
from ding.torch_utils.network.nn_module import fc_block, build_normalization
from ..common import FCEncoder, ConvEncoder


class PCTransformer(nn.Module):
    """
    Overview:
        The transformer block for neural network of algorithms related to Procedure cloning (PC).
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self, cnn_hidden: int, att_hidden: int, att_heads: int, drop_p: float, max_T: int, n_att: int,
            feedforward_hidden: int, n_feedforward: int
    ) -> None:
        """
        Overview:
            Initialize the procedure cloning transformer model according to corresponding input arguments.
        Arguments:
            - cnn_hidden (:obj:`int`): The last channel dimension of CNN encoder, such as 32.
            - att_hidden (:obj:`int`): The dimension of attention blocks, such as 32.
            - att_heads (:obj:`int`): The number of heads in attention blocks, such as 4.
            - drop_p (:obj:`float`): The drop out rate of attention, such as 0.5.
            - max_T (:obj:`int`): The sequence length of procedure cloning, such as 4.
            - n_attn (:obj:`int`): The number of attention layers, such as 4.
            - feedforward_hidden (:obj:`int`):The dimension of feedforward layers, such as 32.
            - n_feedforward (:obj:`int`): The number of feedforward layers, such as 4.
        """
        super().__init__()
        self.n_att = n_att
        self.n_feedforward = n_feedforward
        self.attention_layer = []

        self.norm_layer = [nn.LayerNorm(att_hidden)] * n_att
        self.attention_layer.append(Attention(cnn_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
        for i in range(n_att - 1):
            self.attention_layer.append(Attention(att_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))

        self.att_drop = nn.Dropout(drop_p)

        self.fc_blocks = []
        self.fc_blocks.append(fc_block(att_hidden, feedforward_hidden, activation=nn.ReLU()))
        for i in range(n_feedforward - 1):
            self.fc_blocks.append(fc_block(feedforward_hidden, feedforward_hidden, activation=nn.ReLU()))
        self.norm_layer.extend([nn.LayerNorm(feedforward_hidden)] * n_feedforward)
        self.mask = torch.tril(torch.ones((max_T, max_T), dtype=torch.bool)).view(1, 1, max_T, max_T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            The unique execution (forward) method of PCTransformer.
        Arguments:
            - x (:obj:`torch.Tensor`): Sequential data of several hidden states.
        Returns:
            - output (:obj:`torch.Tensor`): A tensor with the same shape as the input.
        Examples:
            >>> model = PCTransformer(128, 128, 8, 0, 16, 2, 128, 2)
            >>> h = torch.randn((2, 16, 128))
            >>> h = model(h)
            >>> assert h.shape == torch.Size([2, 16, 128])
        """
        for i in range(self.n_att):
            x = self.att_drop(self.attention_layer[i](x, self.mask))
            x = self.norm_layer[i](x)
        for i in range(self.n_feedforward):
            x = self.fc_blocks[i](x)
            x = self.norm_layer[i + self.n_att](x)
        return x


@MODEL_REGISTRY.register('pc_mcts')
class ProcedureCloningMCTS(nn.Module):
    """
    Overview:
        The neural network of algorithms related to Procedure cloning (PC).
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            action_dim: int,
            cnn_hidden_list: SequenceType = [128, 128, 256, 256, 256],
            cnn_activation: nn.Module = nn.ReLU(),
            cnn_kernel_size: SequenceType = [3, 3, 3, 3, 3],
            cnn_stride: SequenceType = [1, 1, 1, 1, 1],
            cnn_padding: SequenceType = [1, 1, 1, 1, 1],
            mlp_hidden_list: SequenceType = [256, 256],
            mlp_activation: nn.Module = nn.ReLU(),
            att_heads: int = 8,
            att_hidden: int = 128,
            n_att: int = 4,
            n_feedforward: int = 2,
            feedforward_hidden: int = 256,
            drop_p: float = 0.5,
            max_T: int = 17
    ) -> None:
        """
        Overview:
            Initialize the MCTS procedure cloning model according to corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Observation space shape, such as [4, 84, 84].
            - action_dim (:obj:`int`): Action space shape, such as 6.
            - cnn_hidden_list (:obj:`SequenceType`): The cnn channel dims for each block, such as\
             [128, 128, 256, 256, 256].
            - cnn_activation (:obj:`nn.Module`): The activation function for cnn blocks, such as ``nn.ReLU()``.
            - cnn_kernel_size (:obj:`SequenceType`): The kernel size for each cnn block, such as [3, 3, 3, 3, 3].
            - cnn_stride (:obj:`SequenceType`): The stride for each cnn block, such as [1, 1, 1, 1, 1].
            - cnn_padding (:obj:`SequenceType`): The padding for each cnn block, such as [1, 1, 1, 1, 1].
            - mlp_hidden_list (:obj:`SequenceType`): The last dim for this must match the last dim of \
            ``cnn_hidden_list``, such as [256, 256].
            - mlp_activation (:obj:`nn.Module`): The activation function for mlp layers, such as ``nn.ReLU()``.
            - att_heads (:obj:`int`): The number of attention heads in transformer, such as 8.
            - att_hidden (:obj:`int`): The number of attention dimension in transformer, such as 128.
            - n_att (:obj:`int`): The number of attention blocks in transformer, such as 4.
            - n_feedforward (:obj:`int`): The number of feedforward layers in transformer, such as 2.
            - drop_p (:obj:`float`): The drop out rate of attention, such as 0.5.
            - max_T (:obj:`int`): The sequence length of procedure cloning, such as 17.
        """
        super().__init__()

        # Conv Encoder
        self.embed_state = ConvEncoder(
            obs_shape, cnn_hidden_list, cnn_activation, cnn_kernel_size, cnn_stride, cnn_padding
        )
        self.embed_action = FCEncoder(action_dim, mlp_hidden_list, activation=mlp_activation)

        self.cnn_hidden_list = cnn_hidden_list

        assert cnn_hidden_list[-1] == mlp_hidden_list[-1]
        layers = []
        for i in range(n_att):
            if i == 0:
                layers.append(Attention(cnn_hidden_list[-1], att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
            else:
                layers.append(Attention(att_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
            layers.append(build_normalization('LN')(att_hidden))
        for i in range(n_feedforward):
            if i == 0:
                layers.append(fc_block(att_hidden, feedforward_hidden, activation=nn.ReLU()))
            else:
                layers.append(fc_block(feedforward_hidden, feedforward_hidden, activation=nn.ReLU()))
                self.layernorm2 = build_normalization('LN')(feedforward_hidden)

        self.transformer = PCTransformer(
            cnn_hidden_list[-1], att_hidden, att_heads, drop_p, max_T, n_att, feedforward_hidden, n_feedforward
        )

        self.predict_goal = torch.nn.Linear(cnn_hidden_list[-1], cnn_hidden_list[-1])
        self.predict_action = torch.nn.Linear(cnn_hidden_list[-1], action_dim)

    def forward(self, states: torch.Tensor, goals: torch.Tensor,
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            ProcedureCloningMCTS forward computation graph, input states tensor and goals tensor, \
            calculate the predicted states and actions.
        Arguments:
            - states (:obj:`torch.Tensor`): The observation of current time.
            - goals (:obj:`torch.Tensor`): The target observation after a period.
            - actions (:obj:`torch.Tensor`): The actions executed during the period.
        Returns:
            - outputs (:obj:`Tuple[torch.Tensor, torch.Tensor]`): Predicted states and actions.
        Examples:
            >>> inputs = { \
                'states': torch.randn(2, 3, 64, 64), \
                'goals': torch.randn(2, 3, 64, 64), \
                'actions': torch.randn(2, 15, 9) \
            }
            >>> model = ProcedureCloningMCTS(obs_shape=(3, 64, 64), action_dim=9)
            >>> goal_preds, action_preds = model(inputs['states'], inputs['goals'], inputs['actions'])
            >>> assert goal_preds.shape == (2, 256)
            >>> assert action_preds.shape == (2, 16, 9)
        """
        B, T, _ = actions.shape

        # shape: (B, h_dim)
        state_embeddings = self.embed_state(states).reshape(B, 1, self.cnn_hidden_list[-1])
        goal_embeddings = self.embed_state(goals).reshape(B, 1, self.cnn_hidden_list[-1])
        # shape: (B, context_len, h_dim)
        actions_embeddings = self.embed_action(actions)

        h = torch.cat((state_embeddings, goal_embeddings, actions_embeddings), dim=1)
        h = self.transformer(h)
        h = h.reshape(B, T + 2, self.cnn_hidden_list[-1])

        goal_preds = self.predict_goal(h[:, 0, :])
        action_preds = self.predict_action(h[:, 1:, :])

        return goal_preds, action_preds


class BFSConvEncoder(nn.Module):
    """
    Overview:
        The ``BFSConvolution Encoder`` used to encode raw 3-dim observations. And output a feature map with the
    same height and width as input. Interfaces: ``__init__``, ``forward``.
    """

    def __init__(
        self,
        obs_shape: SequenceType,
        hidden_size_list: SequenceType = [32, 64, 64, 128],
        activation: Optional[nn.Module] = nn.ReLU(),
        kernel_size: SequenceType = [8, 4, 3],
        stride: SequenceType = [4, 2, 1],
        padding: Optional[SequenceType] = None,
    ) -> None:
        """
        Overview:
            Init the ``BFSConvolution Encoder`` according to the provided arguments.
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
        """
        super(BFSConvEncoder, self).__init__()
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
        layers = layers[:-1]
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output tensor of the env observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Examples:
            >>> model = BFSConvEncoder([3, 16, 16], [32, 32, 4], kernel_size=[3, 3, 3], stride=[1, 1, 1]\
            , padding=[1, 1, 1])
            >>> inputs = torch.randn(3, 16, 16).unsqueeze(0)
            >>> outputs = model(inputs)
            >>> assert outputs['logit'].shape == torch.Size([4, 16, 16])
        """
        return self.main(x)


@MODEL_REGISTRY.register('pc_bfs')
class ProcedureCloningBFS(nn.Module):
    """
    Overview:
        The neural network introduced in procedure cloning (PC) to process 3-dim observations.\
        Given an input, this model will perform several 3x3 convolutions and output a feature map with \
        the same height and width of input. The channel number of output will be the ``action_shape``.
    Interfaces:
         ``__init__``, ``forward``.
    """

    def __init__(
        self,
        obs_shape: SequenceType,
        action_shape: int,
        encoder_hidden_size_list: SequenceType = [128, 128, 256, 256],
    ):
        """
        Overview:
            Init the ``BFSConvolution Encoder`` according to the provided arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, plus one or more ``input size``,\
             such as [4, 84, 84].
            - action_dim (:obj:`int`): Action space shape, such as 6.
            - cnn_hidden_list (:obj:`SequenceType`): The cnn channel dims for each block, such as [128, 128, 256, 256].
        """
        super().__init__()
        num_layers = len(encoder_hidden_size_list)

        kernel_sizes = (3, ) * (num_layers + 1)
        stride_sizes = (1, ) * (num_layers + 1)
        padding_sizes = (1, ) * (num_layers + 1)
        # The output channel equals to action_shape + 1
        encoder_hidden_size_list.append(action_shape + 1)

        self._encoder = BFSConvEncoder(
            obs_shape=obs_shape,
            hidden_size_list=encoder_hidden_size_list,
            kernel_size=kernel_sizes,
            stride=stride_sizes,
            padding=padding_sizes,
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            The computation graph. Given a 3-dim observation, this function will return a tensor with the same \
            height and width. The channel number of output will be the ``action_shape``.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict`): The output dict of model's forward computation graph, \
            only contains a single key ``logit``.
        Examples:
            >>> model = ProcedureCloningBFS([3, 16, 16], 4)
            >>> inputs = torch.randn(16, 16, 3).unsqueeze(0)
            >>> outputs = model(inputs)
            >>> assert outputs['logit'].shape == torch.Size([16, 16, 4])
        """
        x = x.permute(0, 3, 1, 2)
        x = self._encoder(x)
        return {'logit': x.permute(0, 2, 3, 1)}

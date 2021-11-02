from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from ding.torch_utils import get_lstm, one_hot, to_tensor, to_ndarray
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
# from ding.torch_utils.data_helper import one_hot_embedding, one_hot_embedding_none
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead, RainbowHead, \
    QuantileHead, QRDQNHead, DistributionHead


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


@MODEL_REGISTRY.register('ngu')
class NGU(nn.Module):
    """
    Overview:
        DQN + RNN = DRQN
        modified from q_leaning DRQN
        NGU input x_t, a_{t-1}, r_e_{t-1}, r_i_{t-1}, beta
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            collector_env_num: Optional[int] = 1,  # TODO
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
            - lstm_type (:obj:`Optional[str]`): Version of rnn cell, now support ['normal', 'pytorch', 'hpc', 'gru']
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(NGU, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        self.action_shape = action_shape
        self.collector_env_num = collector_env_num
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
        input_size = head_hidden_size + self.collector_env_num + action_shape + 1  # TODO
        # LSTM Type
        self.rnn = get_lstm(lstm_type, input_size=input_size, hidden_size=head_hidden_size)
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

    def forward(
            self, inputs: Dict, inference: bool = False, saved_hidden_state_timesteps: Optional[list] = None
    ) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict DRQN output.
            Parameter updates with DRQN's MLPs forward setup.
        Arguments:
            - inputs (:obj:`Dict`):
            - inference: (:obj:'bool'): if inference is True, we unroll the one timestep transition,
                if inference is False, we unroll the sequence transitions.
            - saved_hidden_state_timesteps: (:obj:'Optional[list]'): when inference is False,
                we unroll the sequence transitions, then we would save rnn hidden states at timesteps
                that are listed in list saved_hidden_state_timesteps.

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
        if 'prev_action' in inputs.keys():
            prev_action = inputs['prev_action']
            prev_reward_e = inputs['prev_reward_e']
        else:
            prev_action = torch.cat(
                [torch.ones_like(inputs['action'][:, 0].unsqueeze(1)) * (-1), inputs['action'][:, :-1]], dim=1
            )  # 20,1  20,31 -> 20,32 self.action_shape
            prev_reward_e = torch.cat(
                [torch.zeros_like(inputs['reward'][:, 0].unsqueeze(1)), inputs['reward'][:, :-1]], dim=1
            )  # 20,1,5  20,31,5 -> 20,32,5
        # prev_action, prev_reward_e,  prev_areward_i =
        # inputs['prev_action'], inputs['prev_reward_e'], inputs['prev_reward_i']
        beta = inputs['beta']  # beta_index
        if inference:  # collect,eval
            x = self.encoder(x)
            x = x.unsqueeze(0)
            prev_reward_e = prev_reward_e.unsqueeze(0).unsqueeze(-1)

            # env_num= x.shape[1] # collect 8 eval 5
            env_num = self.collector_env_num
            # env_num = 8
            beta_onehot = one_hot(beta, env_num).unsqueeze(0)
            prev_action_onehot = one_hot(prev_action, self.action_shape).unsqueeze(0)
            x_a_r_beta = torch.cat([x, prev_action_onehot, prev_reward_e, beta_onehot], dim=-1)  # shape [1,8,80]
            x, next_state = self.rnn(x_a_r_beta.to(torch.float32), prev_state)
            # x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)
            x = self.head(x)
            x['next_state'] = next_state
            return x
        else:  # train
            assert len(x.shape) in [3, 5], x.shape  # 20,32,2739
            x = parallel_wrapper(self.encoder)(x)  # 20,32,64
            # if nstep:
            prev_reward_e = prev_reward_e[:, :, 0].unsqueeze(-1)  # 20,32,1
            env_num = self.collector_env_num
            beta_onehot = one_hot(beta.view(-1), env_num).view([beta.shape[0], beta.shape[1], -1])  # 20,32,8
            prev_action_onehot = one_hot(prev_action.view(-1), self.action_shape).view(
                [prev_action.shape[0], prev_action.shape[1], -1]
            )  # 20,32,7
            x_a_r_beta = torch.cat([x, prev_action_onehot, prev_reward_e, beta_onehot], dim=-1)  # 20,32,80
            x = x_a_r_beta
            lstm_embedding = []
            # TODO(nyz) how to deal with hidden_size key-value
            hidden_state_list = []
            if saved_hidden_state_timesteps is not None:
                saved_hidden_state = []
            for t in range(x.shape[0]):  # T timesteps
                output, prev_state = self.rnn(x[t:t + 1], prev_state)
                if saved_hidden_state_timesteps is not None and t + 1 in saved_hidden_state_timesteps:
                    saved_hidden_state.append(prev_state)
                lstm_embedding.append(output)
                hidden_state = list(zip(*prev_state))
                hidden_state_list.append(torch.cat(hidden_state[0], dim=1))  # take the first hidden state
            x = torch.cat(lstm_embedding, 0)  # [20, 32, 64]
            x = parallel_wrapper(self.head)(x)
            x['next_state'] = prev_state  # including the first hidden state and the second cell state
            x['hidden_state'] = torch.cat(hidden_state_list, dim=-3)
            if saved_hidden_state_timesteps is not None:
                x['saved_hidden_state'] = saved_hidden_state  # the selected saved hidden states
            return x


# @MODEL_REGISTRY.register('ngunoar')
# class NGUnoar(nn.Module):
#     """
#     Overview:
#         DQN + RNN = DRQN
#         modified from q_leaning DRQN
#         NGU input x_t, a_{t-1}, r_e_{t-1}, r_i_{t-1}, beta
#     """
#
#     def __init__(
#             self,
#             obs_shape: Union[int, SequenceType],
#             action_shape: Union[int, SequenceType],
#             encoder_hidden_size_list: SequenceType = [128, 128, 64],
#             dueling: bool = True,
#             head_hidden_size: Optional[int] = None,
#             head_layer_num: int = 1,
#             lstm_type: Optional[str] = 'normal',
#             activation: Optional[nn.Module] = nn.ReLU(),
#             norm_type: Optional[str] = None
#     ) -> None:
#         r"""
#         Overview:
#             Init the DRQN Model according to arguments.
#         Arguments:
#             - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
#             - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
#             - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
#             - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
#             - lstm_type (:obj:`Optional[str]`): Version of rnn cell, now support ['normal', 'pytorch', 'hpc', 'gru']
#             - activation (:obj:`Optional[nn.Module]`):
#                 The type of activation function to use in ``MLP`` the after ``layer_fn``,
#                 if ``None`` then default set to ``nn.ReLU()``
#             - norm_type (:obj:`Optional[str]`):
#                 The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
#         """
#         super(NGUnoar, self).__init__()
#         # For compatibility: 1, (1, ), [4, 32, 32]
#         obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
#         if head_hidden_size is None:
#             head_hidden_size = encoder_hidden_size_list[-1]
#         # FC Encoder
#         if isinstance(obs_shape, int) or len(obs_shape) == 1:
#             self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
#         # Conv Encoder
#         elif len(obs_shape) == 3:
#             self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, \
#             norm_type=norm_type)
#         else:
#             raise RuntimeError(
#                 "not support obs_shape for pre-defined encoder: {}, please customize your own DRQN".format(obs_shape)
#             )
#         input_size = head_hidden_size + 8
#         # LSTM Type
#         self.rnn = get_lstm(lstm_type, input_size=input_size, hidden_size=head_hidden_size)
#         # Head Type
#         if dueling:
#             head_cls = DuelingHead
#         else:
#             head_cls = DiscreteHead
#         multi_head = not isinstance(action_shape, int)
#         if multi_head:
#             self.head = MultiHead(
#                 head_cls,
#                 head_hidden_size,
#                 action_shape,
#                 layer_num=head_layer_num,
#                 activation=activation,
#                 norm_type=norm_type
#             )
#         else:
#             self.head = head_cls(
#                 head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
#             )
#
#     def forward(self, inputs: Dict, inference: bool = False) -> Dict:
#         r"""
#         Overview:
#             Use observation tensor to predict DRQN output.
#             Parameter updates with DRQN's MLPs forward setup.
#         Arguments:
#             - inputs (:obj:`Dict`):
#
#        ArgumentsKeys:
#             - obs (:obj:`torch.Tensor`): Encoded observation
#             - prev_state (:obj:`list`): Previous state's tensor of size ``(B, N)``
#
#         Returns:
#             - outputs (:obj:`Dict`):
#                 Run ``MLP`` with ``DRQN`` setups and return the result prediction dictionary.
#
#         ReturnsKeys:
#             - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``obs``.
#             - next_state (:obj:`list`): Next state's tensor of size ``(B, N)``
#         Shapes:
#             - obs (:obj:`torch.Tensor`): :math:`(B, N=obs_space)`, where B is batch size.
#             - prev_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`
#             - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`
#             - next_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`
#
#         Examples:
#             >>> # Init input's Keys:
#             >>> prev_state = [[torch.randn(1, 1, 64) for __ in range(2)] for _ in range(4)] # B=4
#             >>> obs = torch.randn(4,64)
#             >>> model = DRQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
#             >>> outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=True)
#             >>> # Check outputs's Keys
#             >>> assert isinstance(outputs, dict)
#             >>> assert outputs['logit'].shape == (4, 64)
#             >>> assert len(outputs['next_state']) == 4
#             >>> assert all([len(t) == 2 for t in outputs['next_state']])
#             >>> assert all([t[0].shape == (1, 1, 64) for t in outputs['next_state']])
#         """
#
#         x, prev_state = inputs['obs'], inputs['prev_state']
#         # prev_action, prev_reward_e,  prev_areward_i = \
#         # inputs['prev_action'], inputs['prev_reward_e'], inputs['prev_reward_i']
#         beta = inputs['beta']  # beta_index
#         if inference:
#             x = self.encoder(x)
#             x = x.unsqueeze(0)
#             # x_a_r_beta = torch.cat(x, prev_action, prev_reward_e, prev_reward_i, beta)
#             # beta=beta.unsqueeze(1).unsqueeze(0)
#             # env_num= x.shape[1] #collect 8 eval 5
#             env_num = 8
#             beta = one_hot_embedding(beta, env_num).unsqueeze(0)
#             x_beta = torch.cat([x, beta], dim=-1)
#             x, next_state = self.rnn(x_beta, prev_state)
#             # x, next_state = self.rnn(x, prev_state)
#             x = x.squeeze(0)
#             x = self.head(x)
#             x['next_state'] = next_state
#             return x
#         else:
#             assert len(x.shape) in [3, 5], x.shape  # 20,32,2739
#             x = parallel_wrapper(self.encoder)(x)  # 20,32,64
#
#             env_num = 8  # x.shape[1]
#             beta = one_hot_embedding(beta, env_num)
#             x_beta = torch.cat([x, beta], dim=-1)
#             x = x_beta
#             lstm_embedding = []
#             # TODO(nyz) how to deal with hidden_size key-value
#             hidden_state_list = []
#             for t in range(x.shape[0]):  # T timesteps
#                 output, prev_state = self.rnn(x[t:t + 1], prev_state)
#                 lstm_embedding.append(output)
#                 hidden_state = list(zip(*prev_state))
#                 hidden_state_list.append(torch.cat(hidden_state[0], dim=1))
#             x = torch.cat(lstm_embedding, 0)
#             x = parallel_wrapper(self.head)(x)
#             x['next_state'] = prev_state
#             x['hidden_state'] = torch.cat(hidden_state_list, dim=-3)
#             return x

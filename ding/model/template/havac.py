from typing import Union, Dict, Optional
import torch
import torch.nn as nn

from ding.torch_utils import get_lstm
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ding.model.template.q_learning import parallel_wrapper
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, \
    FCEncoder, ConvEncoder


class RNNLayer(nn.Module):

    def __init__(self, lstm_type, input_size, hidden_size, res_link: bool = False):
        super(RNNLayer, self).__init__()
        self.rnn = get_lstm(lstm_type, input_size=input_size, hidden_size=hidden_size)
        self.res_link = res_link

    def forward(self, x, prev_state, inference: bool = False):
        """
        Forward pass of the RNN layer.
        If inference is True, sequence length of input is set to 1.
        If res_link is True, a residual link is added to the output.
        """
        # x: obs_embedding
        if self.res_link:
            a = x
        if inference:
            x = x.unsqueeze(0)  # for rnn input, put the seq_len of x as 1 instead of none.
            # prev_state: DataType: List[Tuple[torch.Tensor]]; Initially, it is a list of None
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)  # to delete the seq_len dim to match head network input
            if self.res_link:
                x = x + a
            return {'output': x, 'next_state': next_state}
        else:
            # lstm_embedding stores all hidden_state
            lstm_embedding = []
            hidden_state_list = []
            for t in range(x.shape[0]):  # T timesteps
                # use x[t:t+1] but not x[t] can keep original dimension
                output, prev_state = self.rnn(x[t:t + 1], prev_state)  # output: (1,B, head_hidden_size)
                lstm_embedding.append(output)
                hidden_state = [p['h'] for p in prev_state]
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
                hidden_state_list.append(torch.cat(hidden_state, dim=1))
            x = torch.cat(lstm_embedding, 0)  # (T, B, head_hidden_size)
            if self.res_link:
                x = x + a
            all_hidden_state = torch.cat(hidden_state_list, dim=0)
            return {'output': x, 'next_state': prev_state, 'hidden_state': all_hidden_state}


@MODEL_REGISTRY.register('havac')
class HAVAC(nn.Module):
    """
    Overview:
        The HAVAC model of each agent for HAPPO.
    Interfaces:
        ``__init__``, ``forward``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        agent_obs_shape: Union[int, SequenceType],
        global_obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        agent_num: int,
        use_lstm: bool = False,
        lstm_type: str = 'gru',
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 2,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        action_space: str = 'discrete',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        bound_type: Optional[str] = None,
        res_link: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the VAC Model for HAPPO according to arguments.
        Arguments:
            - agent_obs_shape (:obj:`Union[int, SequenceType]`): Observation's space for single agent.
            - global_obs_shape (:obj:`Union[int, SequenceType]`): Observation's space for global agent
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - agent_num (:obj:`int`): Number of agents.
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for critic's nn.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
            - res_link (:obj:`bool`): use the residual link or not, default to False
        """
        super(HAVAC, self).__init__()
        self.agent_num = agent_num
        self.agent_models = nn.ModuleList(
            [
                HAVACAgent(
                    agent_obs_shape=agent_obs_shape,
                    global_obs_shape=global_obs_shape,
                    action_shape=action_shape,
                    use_lstm=use_lstm,
                    action_space=action_space,
                ) for _ in range(agent_num)
            ]
        )

    def forward(self, agent_idx, input_data, mode):
        selected_agent_model = self.agent_models[agent_idx]
        output = selected_agent_model(input_data, mode)
        return output


class HAVACAgent(nn.Module):
    """
    Overview:
        The HAVAC model of each agent for HAPPO.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``, ``compute_actor_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        agent_obs_shape: Union[int, SequenceType],
        global_obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        use_lstm: bool = False,
        lstm_type: str = 'gru',
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 2,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        action_space: str = 'discrete',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'happo',
        bound_type: Optional[str] = None,
        res_link: bool = False,
    ) -> None:
        r"""
        Overview:
            Init the VAC Model for HAPPO according to arguments.
        Arguments:
            - agent_obs_shape (:obj:`Union[int, SequenceType]`): Observation's space for single agent.
            - global_obs_shape (:obj:`Union[int, SequenceType]`): Observation's space for global agent
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for critic's nn.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
            - res_link (:obj:`bool`): use the residual link or not, default to False
        """
        super(HAVACAgent, self).__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape, self.agent_obs_shape, self.action_shape = global_obs_shape, agent_obs_shape, action_shape
        self.action_space = action_space
        # Encoder Type
        if isinstance(agent_obs_shape, int) or len(agent_obs_shape) == 1:
            actor_encoder_cls = FCEncoder
        elif len(agent_obs_shape) == 3:
            actor_encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own VAC".
                format(agent_obs_shape)
            )
        if isinstance(global_obs_shape, int) or len(global_obs_shape) == 1:
            critic_encoder_cls = FCEncoder
        elif len(global_obs_shape) == 3:
            critic_encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own VAC".
                format(global_obs_shape)
            )

        # We directly connect the Head after a Liner layer instead of using the 3-layer FCEncoder.
        # In SMAC task it can obviously improve the performance.
        # Users can change the model according to their own needs.
        self.actor_encoder = actor_encoder_cls(
            obs_shape=agent_obs_shape,
            hidden_size_list=encoder_hidden_size_list,
            activation=activation,
            norm_type=norm_type
        )
        self.critic_encoder = critic_encoder_cls(
            obs_shape=global_obs_shape,
            hidden_size_list=encoder_hidden_size_list,
            activation=activation,
            norm_type=norm_type
        )
        # RNN part
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.actor_rnn = RNNLayer(
                lstm_type,
                input_size=encoder_hidden_size_list[-1],
                hidden_size=actor_head_hidden_size,
                res_link=res_link
            )
            self.critic_rnn = RNNLayer(
                lstm_type,
                input_size=encoder_hidden_size_list[-1],
                hidden_size=critic_head_hidden_size,
                res_link=res_link
            )
        # Head Type
        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )
        assert self.action_space in ['discrete', 'continuous'], self.action_space
        if self.action_space == 'discrete':
            self.actor_head = DiscreteHead(
                actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type
            )
        elif self.action_space == 'continuous':
            self.actor_head = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type
            )
        # must use list, not nn.ModuleList
        self.actor = [self.actor_encoder, self.actor_rnn, self.actor_head] if self.use_lstm \
            else [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_rnn, self.critic_head] if self.use_lstm \
            else [self.critic_encoder, self.critic_head]
        # for convenience of call some apis(such as: self.critic.parameters()), but may cause
        # misunderstanding when print(self)
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict output.
            Parameter updates with VAC's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'`` or ``'compute_critic'``:
                - inputs (:obj:`torch.Tensor`):
                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

                Forward with ``'compute_actor'``, Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.

                Forward with ``'compute_critic'``, Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N corresponding ``hidden_size``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Actor Examples:
            >>> model = VAC(64,128)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 128])

        Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> critic_outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)

        Actor-Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: Dict, inference: bool = False) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                input data dict with keys ['obs'(with keys ['agent_state', 'global_state', 'action_mask']),
                  'actor_prev_state']
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder RNN(optional) and head.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor.
            - actor_next_state:
            - hidden_state
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - actor_next_state: (B,)
            - hidden_state:

        Examples:
            >>> model = HAVAC(
                    agent_obs_shape=obs_dim,
                    global_obs_shape=global_obs_dim,
                    action_shape=action_dim,
                    use_lstm = True,
                    )
            >>> inputs = {
                    'obs': {
                        'agent_state': torch.randn(T, bs, obs_dim),
                        'global_state': torch.randn(T, bs, global_obs_dim),
                        'action_mask': torch.randint(0, 2, size=(T, bs, action_dim))
                    },
                    'actor_prev_state': [None for _ in range(bs)],
                }
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == (T, bs, action_dim)
        """
        x = inputs['obs']['agent_state']
        output = {}
        if self.use_lstm:
            rnn_actor_prev_state = inputs['actor_prev_state']
            if inference:
                x = self.actor_encoder(x)
                rnn_output = self.actor_rnn(x, rnn_actor_prev_state, inference)
                x = rnn_output['output']
                x = self.actor_head(x)
                output['next_state'] = rnn_output['next_state']
                # output: 'logit'/'next_state'
            else:
                assert len(x.shape) in [3, 5], x.shape
                x = parallel_wrapper(self.actor_encoder)(x)  # (T, B, N)
                rnn_output = self.actor_rnn(x, rnn_actor_prev_state, inference)
                x = rnn_output['output']
                x = parallel_wrapper(self.actor_head)(x)
                output['actor_next_state'] = rnn_output['next_state']
                output['actor_hidden_state'] = rnn_output['hidden_state']
                # output: 'logit'/'actor_next_state'/'hidden_state'
        else:
            x = self.actor_encoder(x)
            x = self.actor_head(x)
            # output: 'logit'

        if self.action_space == 'discrete':
            action_mask = inputs['obs']['action_mask']
            logit = x['logit']
            logit[action_mask == 0.0] = -99999999
        elif self.action_space == 'continuous':
            logit = x
        output['logit'] = logit
        return output

    def compute_critic(self, inputs: Dict, inference: bool = False) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`Dict`):
                input data dict with keys ['obs'(with keys ['agent_state', 'global_state', 'action_mask']),
                  'critic_prev_state'(when you are using rnn)]
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder [rnn] and head.

                Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
                    - logits
        Shapes:
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.
            - logits

        Examples:
            >>> model = HAVAC(
                    agent_obs_shape=obs_dim,
                    global_obs_shape=global_obs_dim,
                    action_shape=action_dim,
                    use_lstm = True,
                    )
            >>> inputs = {
                    'obs': {
                        'agent_state': torch.randn(T, bs, obs_dim),
                        'global_state': torch.randn(T, bs, global_obs_dim),
                        'action_mask': torch.randint(0, 2, size=(T, bs, action_dim))
                    },
                    'critic_prev_state': [None for _ in range(bs)],
                }
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> assert critic_outputs['value'].shape == (T, bs))
        """
        global_obs = inputs['obs']['global_state']
        output = {}
        if self.use_lstm:
            rnn_critic_prev_state = inputs['critic_prev_state']
            if inference:
                x = self.critic_encoder(global_obs)
                rnn_output = self.critic_rnn(x, rnn_critic_prev_state, inference)
                x = rnn_output['output']
                x = self.critic_head(x)
                output['next_state'] = rnn_output['next_state']
                # output: 'value'/'next_state'
            else:
                assert len(global_obs.shape) in [3, 5], global_obs.shape
                x = parallel_wrapper(self.critic_encoder)(global_obs)  # (T, B, N)
                rnn_output = self.critic_rnn(x, rnn_critic_prev_state, inference)
                x = rnn_output['output']
                x = parallel_wrapper(self.critic_head)(x)
                output['critic_next_state'] = rnn_output['next_state']
                output['critic_hidden_state'] = rnn_output['hidden_state']
                # output: 'value'/'critic_next_state'/'hidden_state'
        else:
            x = self.critic_encoder(global_obs)
            x = self.critic_head(x)
            # output: 'value'
        output['value'] = x['pred']
        return output

    def compute_actor_critic(self, inputs: Dict, inference: bool = False) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:dict): input data dict with keys
                ['obs'(with keys ['agent_state', 'global_state', 'action_mask']),
                'actor_prev_state', 'critic_prev_state'(when you are using rnn)]

        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.
            - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])


        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder.
            Returning the combination dictionry.

        """
        actor_output = self.compute_actor(inputs, inference)
        critic_output = self.compute_critic(inputs, inference)
        if self.use_lstm:
            return {
                'logit': actor_output['logit'],
                'value': critic_output['value'],
                'actor_next_state': actor_output['actor_next_state'],
                'actor_hidden_state': actor_output['actor_hidden_state'],
                'critic_next_state': critic_output['critic_next_state'],
                'critic_hidden_state': critic_output['critic_hidden_state'],
            }
        else:
            return {
                'logit': actor_output['logit'],
                'value': critic_output['value'],
            }

from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


@MODEL_REGISTRY.register('qac')
class QAC(nn.Module):
    r"""
    Overview:
        The QAC network, which is used in DDPG/TD3/SAC.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType, EasyDict],
            action_space: str,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        """
        Overview:
            Initailize the QAC Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's shape, such as 128, (156, ).
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's shape, such as 4, (3, ), \
                EasyDict({'action_type_shape': 3, 'action_args_shape': 4}).
            - action_space (:obj:`str`): The type of action space, \
                including [``regression``, ``reparameterization``, ``hybrid``].
            - twin_critic (:obj:`bool`): Whether to use twin critic, one of tricks in TD3.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor head.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for actor head.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic head.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for critic head.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` \
                after each FC layer, if ``None`` then default set to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization to after network layer (FC, Conv), \
                see ``ding.torch_utils.network`` for more details.
        """
        super(QAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization', 'hybrid']
        if self.action_space == 'regression':  # DDPG, TD3
            self.actor = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                RegressionHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.action_space == 'reparameterization':  # SAC
            self.actor = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                ReparameterizationHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    sigma_type='conditioned',
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.action_space == 'hybrid':  # PADDPG
            # hybrid action space: action_type(discrete) + action_args(continuous),
            # such as {'action_type_shape': torch.LongTensor([0]), 'action_args_shape': torch.FloatTensor([0.1, -0.27])}
            action_shape.action_args_shape = squeeze(action_shape.action_args_shape)
            action_shape.action_type_shape = squeeze(action_shape.action_type_shape)
            actor_action_args = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                RegressionHead(
                    actor_head_hidden_size,
                    action_shape.action_args_shape,
                    actor_head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
            actor_action_type = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                DiscreteHead(
                    actor_head_hidden_size,
                    action_shape.action_type_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type,
                )
            )
            self.actor = nn.ModuleList([actor_action_type, actor_action_args])

        self.twin_critic = twin_critic
        if self.action_space == 'hybrid':
            critic_input_size = obs_shape + action_shape.action_type_shape + action_shape.action_args_shape
        else:
            critic_input_size = obs_shape + action_shape
        if self.twin_critic:
            self.critic = nn.ModuleList()
            for _ in range(2):
                self.critic.append(
                    nn.Sequential(
                        nn.Linear(critic_input_size, critic_head_hidden_size), activation,
                        RegressionHead(
                            critic_head_hidden_size,
                            1,
                            critic_head_layer_num,
                            final_tanh=False,
                            activation=activation,
                            norm_type=norm_type
                        )
                    )
                )
        else:
            self.critic = nn.Sequential(
                nn.Linear(critic_input_size, critic_head_hidden_size), activation,
                RegressionHead(
                    critic_head_hidden_size,
                    1,
                    critic_head_layer_num,
                    final_tanh=False,
                    activation=activation,
                    norm_type=norm_type
                )
            )

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], mode: str) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The unique execution (forward) method of QAC method, and one can indicate different modes to implement \
            different computation graph, including ``compute_actor`` and ``compute_critic`` in QAC.
        Mode compute_actor:
            Arguments:
                - inputs (:obj:`torch.Tensor`): Observation data, defaults to tensor.
            Returns:
                - output (:obj:`Dict`): Output dict data, including differnet key-values among distinct action_space.
        Mode compute_critic:
            Arguments:
                - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
            Returns:
                - output (:obj:`Dict`): Output dict data, including q_value tensor.

        .. note::
            For specific examples, one can refer to API doc of ``compute_actor`` and ``compute_critic`` respectively.
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, obs: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Overview:
            The forward computation graph of compute_actor mode, uses observation tensor to produce actor output,
            such as ``action``, ``logit`` and so on.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observation tensor data, now supports a batch of 1-dim vector data, \
                i.e. ``(B, obs_shape)``.
        Returns:
            - outputs (:obj:`Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]`): Actor output varying \
                from action_space: ``regression``, ``reparameterization``, ``hybrid``.

        ReturnsKeys (either):
            - regression action_space
                - action (:obj:`torch.Tensor`): Continuous action with same size as ``action_shape``, usually in DDPG.
            - reparameterization action_space
                - logit (:obj:`Dict[str, torch.Tensor]`): Reparameterization logit, usually in SAC.

                    - mu (:obj:`torch.Tensor`): Mean of parameterization gaussion distribution.
                    - sigma (:obj:`torch.Tensor`): Standard variation of parameterization gaussion distribution.
            - hybrid action_space
                - logit (:obj:`torch.Tensor`): Discrete action type logit.
                - action_args (:obj:`torch.Tensor`): Continuous action arguments.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``obs_shape``.
            - action (:obj:`torch.Tensor`): :math:`(B, N1)`, B is batch size and N1 corresponds to ``action_shape``.
            - logit.mu (:obj:`torch.Tensor`): :math:`(B, N1)`, B is batch size and N1 corresponds to ``action_shape``.
            - logit.sigma (:obj:`torch.Tensor`): :math:`(B, N1)`, B is batch size.
            - logit (:obj:`torch.Tensor`): :math:`(B, N2)`, B is batch size and N2 corresponds to \
                ``action_shape.action_type_shape``.
            - action_args (:obj:`torch.Tensor`): :math:`(B, N3)`, B is batch size and N3 corresponds to \
                ``action_shape.action_args_shape``.
        Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['logit'][0].shape == torch.Size([4, 64])  # mu
            >>> actor_outputs['logit'][1].shape == torch.Size([4, 64]) # sigma
        """
        if self.action_space == 'regression':
            x = self.actor(obs)
            return {'action': x['pred']}
        elif self.action_space == 'reparameterization':
            x = self.actor(obs)
            return {'logit': [x['mu'], x['sigma']]}
        elif self.action_space == 'hybrid':
            logit = self.actor[0](obs)
            action_args = self.actor[1](obs)
            return {'logit': logit['logit'], 'action_args': action_args['pred']}

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The forward computation graph of compute_critic mode, uses observation and action tensor to produce critic
            output, such as ``q_value``.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): Dict strcture of input data, including ``obs`` and ``action`` \
                tensor, also contains ``logit`` tensor in hybrid action_space.
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): Critic output, such as ``q_value``.

        ArgumentsKeys:
            - obs: (:obj:`torch.Tensor`): Observation tensor data, now supports a batch of 1-dim vector data.
            - action (:obj:`Union[torch.Tensor, Dict]`): Continuous action with same size as ``action_shape``.
            - logit (:obj:`torch.Tensor`): Discrete action logit, only in hybrid action_space.
            - action_args (:obj:`torch.Tensor`): Continuous action arguments, only in hybrid action_space.
        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``.
            - logit (:obj:`torch.Tensor`): :math:`(B, N2)`, B is batch size and N2 corresponds to \
                ``action_shape.action_type_shape``.
            - action_args (:obj:`torch.Tensor`): :math:`(B, N3)`, B is batch size and N3 corresponds to \
                ``action_shape.action_args_shape``.
            - action (:obj:`torch.Tensor`): :math:`(B, N4)`, where B is batch size and N4 is ``action_shape``.
            - q_value (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> inputs = {'obs': torch.randn(4, 8), 'action': torch.randn(4, 1)}
            >>> model = QAC(obs_shape=(8, ),action_shape=1, action_space='regression')
            >>> model(inputs, mode='compute_critic')['q_value']  # q value
            ... tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)
        """

        obs, action = inputs['obs'], inputs['action']
        assert len(obs.shape) == 2
        if self.action_space == 'hybrid':
            action_type_logit = inputs['logit']
            action_type_logit = torch.softmax(action_type_logit, dim=-1)
            action_args = action['action_args']
            if len(action_args.shape) == 1:
                action_args = action_args.unsqueeze(1)
            x = torch.cat([obs, action_type_logit, action_args], dim=1)
        else:
            if len(action.shape) == 1:  # (B, ) -> (B, 1)
                action = action.unsqueeze(1)
            x = torch.cat([obs, action], dim=1)
        if self.twin_critic:
            x = [m(x)['pred'] for m in self.critic]
        else:
            x = self.critic(x)['pred']
        return {'q_value': x}


@MODEL_REGISTRY.register('discrete_qac')
class DiscreteQAC(nn.Module):
    r"""
    Overview:
        The Discrete QAC model, used in DiscreteSAC.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [64],
            twin_critic: bool = False,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the QAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - twin_critic (:obj:`bool`): Whether include twin critic.
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
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details.
        """
        super(DiscreteQAC, self).__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        action_shape: int = squeeze(action_shape)

        if isinstance(agent_obs_shape, int) or len(agent_obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(agent_obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".
                format(agent_obs_shape)
            )
        if isinstance(global_obs_shape, int) or len(global_obs_shape) == 1:
            global_encoder_cls = FCEncoder
        elif len(global_obs_shape) == 3:
            global_encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".
                format(global_obs_shape)
            )

        self.actor = nn.Sequential(
            encoder_cls(agent_obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type),
            DiscreteHead(
                actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type
            )
        )

        self.twin_critic = twin_critic
        if self.twin_critic:
            self.critic = nn.ModuleList()
            for _ in range(2):
                self.critic.append(
                    nn.Sequential(
                        global_encoder_cls(
                            agent_obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
                        ),
                        DiscreteHead(
                            critic_head_hidden_size,
                            action_shape,
                            critic_head_layer_num,
                            activation=activation,
                            norm_type=norm_type
                        )
                    )
                )
        else:
            self.critic = nn.Sequential(
                global_encoder_cls(
                    agent_obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
                ),
                DiscreteHead(
                    critic_head_hidden_size,
                    action_shape,
                    critic_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            )

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use bbservation and action tensor to predict output.
            Parameter updates with QAC's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'``:
                - inputs (:obj:`torch.Tensor`):
                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.

            Forward with ``'compute_critic'``, inputs (`Dict`) Necessary Keys:
                - ``obs``, ``action`` encoded tensors.

            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.

                Forward with ``'compute_actor'``, Necessary Keys (either):
                    - action (:obj:`torch.Tensor`): Action tensor with same size as input ``x``.
                    - logit (:obj:`torch.Tensor`):
                        Logit tensor encoding ``mu`` and ``sigma``, both with same size as input ``x``.

                Forward with ``'compute_critic'``, Necessary Keys:
                    - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Actor Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Critic Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is``action_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N3 is ``action_shape``

        Actor Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> actor_outputs['logit'][0].shape # mu
            >>> torch.Size([4, 64])
            >>> actor_outputs['logit'][1].shape # sigma
            >>> torch.Size([4, 64])

        Critic Examples:
            >>> inputs = {'obs': torch.randn(4,N), 'action': torch.randn(4,1)}
            >>> model = QAC(obs_shape=(N, ), action_shape=1, action_space='regression')
            >>> model(inputs, mode='compute_critic')['q_value'] # q value
            tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict output.
            Execute parameter updates with ``'compute_actor'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = actor_head_hidden_size``
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of forward pass encoder and head.

        ReturnsKeys (either):
            - action (:obj:`torch.Tensor`): Continuous action tensor with same size as ``action_shape``.
            - logit (:obj:`torch.Tensor`):
                Logit tensor encoding ``mu`` and ``sigma``, both with same size as input ``x``.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - logit (:obj:`list`): 2 elements, mu and sigma, each is the shape of :math:`(B, N0)`.
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, B is batch size.
        Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> actor_outputs['logit'][0].shape # mu
            >>> torch.Size([4, 64])
            >>> actor_outputs['logit'][1].shape # sigma
            >>> torch.Size([4, 64])
        """
        x = self.actor(inputs['obs'])
        return {'logit': x['logit']}

    def compute_critic(self, inputs: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - ``obs``, ``action`` encoded tensors.
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Q-value output.

        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> inputs = {'obs': torch.randn(4, N), 'action': torch.randn(4, 1)}
            >>> model = QAC(obs_shape=(N, ),action_shape=1, action_space='regression')
            >>> model(inputs, mode='compute_critic')['q_value'] # q value
            tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)

        """

        if self.twin_critic:
            x = [m(inputs['obs'])['logit'] for m in self.critic]
        else:
            x = self.critic(inputs['obs'])['logit']
        return {'q_value': x}

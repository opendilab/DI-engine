from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


@MODEL_REGISTRY.register('continuous_qac')
class ContinuousQAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to Q-value Actor-Critic (QAC), such as \
        DDPG/TD3/SAC. This model now supports continuous and hybrid action space. The ContinuousQAC is composed of \
        four parts: ``actor_encoder``, ``critic_encoder``, ``actor_head`` and ``critic_head``. Encoders are used to \
        extract the feature from various observation. Heads are used to predict corresponding Q-value or action logit. \
        In high-dimensional observation space like 2D image, we often use a shared encoder for both ``actor_encoder`` \
        and ``critic_encoder``. In low-dimensional observation space like 1D vector, we often use different encoders.
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
        encoder_hidden_size_list: Optional[SequenceType] = None,
        share_encoder: Optional[bool] = False,
    ) -> None:
        """
        Overview:
            Initailize the ContinuousQAC Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's shape, such as 128, (156, ).
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's shape, such as 4, (3, ), \
                EasyDict({'action_type_shape': 3, 'action_args_shape': 4}).
            - action_space (:obj:`str`): The type of action space, including [``regression``, ``reparameterization``, \
                ``hybrid``], ``regression`` is used for DDPG/TD3, ``reparameterization`` is used for SAC and \
                ``hybrid`` for PADDPG.
            - twin_critic (:obj:`bool`): Whether to use twin critic, one of tricks in TD3.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor head.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the actor network to compute action.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic head.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the critic network to compute Q-value.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` \
                after each FC layer, if ``None`` then default set to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization to after network layer (FC, Conv), \
                see ``ding.torch_utils.network`` for more details.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``, this argument is only used in image observation.
            - share_encoder (:obj:`Optional[bool]`): Whether to share encoder between actor and critic.
        """
        super(ContinuousQAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization', 'hybrid'], self.action_space

        # encoder
        self.share_encoder = share_encoder
        if np.isscalar(obs_shape) or len(obs_shape) == 1:
            assert not self.share_encoder, "Vector observation doesn't need share encoder."
            assert encoder_hidden_size_list is None, "Vector obs encoder only uses one layer nn.Linear"
            # Because there is already a layer nn.Linear in the head, so we use nn.Identity here to keep
            # compatible with the image observation and avoid adding an extra layer nn.Linear.
            self.actor_encoder = nn.Identity()
            self.critic_encoder = nn.Identity()
            encoder_output_size = obs_shape
        elif len(obs_shape) == 3:

            def setup_conv_encoder():
                kernel_size = [3 for _ in range(len(encoder_hidden_size_list))]
                stride = [2] + [1 for _ in range(len(encoder_hidden_size_list) - 1)]
                return ConvEncoder(
                    obs_shape,
                    encoder_hidden_size_list,
                    activation=activation,
                    norm_type=norm_type,
                    kernel_size=kernel_size,
                    stride=stride
                )

            if self.share_encoder:
                encoder = setup_conv_encoder()
                self.actor_encoder = self.critic_encoder = encoder
            else:
                self.actor_encoder = setup_conv_encoder()
                self.critic_encoder = setup_conv_encoder()
            encoder_output_size = self.actor_encoder.output_size
        else:
            raise RuntimeError("not support observation shape: {}".format(obs_shape))
        # head
        if self.action_space == 'regression':  # DDPG, TD3
            self.actor_head = nn.Sequential(
                nn.Linear(encoder_output_size, actor_head_hidden_size), activation,
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
            self.actor_head = nn.Sequential(
                nn.Linear(encoder_output_size, actor_head_hidden_size), activation,
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
                nn.Linear(encoder_output_size, actor_head_hidden_size), activation,
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
                nn.Linear(encoder_output_size, actor_head_hidden_size), activation,
                DiscreteHead(
                    actor_head_hidden_size,
                    action_shape.action_type_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type,
                )
            )
            self.actor_head = nn.ModuleList([actor_action_type, actor_action_args])

        self.twin_critic = twin_critic
        if self.action_space == 'hybrid':
            critic_input_size = encoder_output_size + action_shape.action_type_shape + action_shape.action_args_shape
        else:
            critic_input_size = encoder_output_size + action_shape
        if self.twin_critic:
            self.critic_head = nn.ModuleList()
            for _ in range(2):
                self.critic_head.append(
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
            self.critic_head = nn.Sequential(
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

        # Convenient for calling some apis (e.g. self.critic.parameters()),
        # but may cause misunderstanding when `print(self)`
        self.actor = nn.ModuleList([self.actor_encoder, self.actor_head])
        self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], mode: str) -> Dict[str, torch.Tensor]:
        """
        Overview:
            QAC forward computation graph, input observation tensor to predict Q-value or action logit. Different \
            ``mode`` will forward with different network modules to get different outputs and save computation.
        Arguments:
            - inputs (:obj:`Union[torch.Tensor, Dict[str, torch.Tensor]]`): The input data for forward computation \
                graph, for ``compute_actor``, it is the observation tensor, for ``compute_critic``, it is the \
                dict data including obs and action tensor.
            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of QAC forward computation graph, whose \
                key-values vary in different forward modes.
        Examples (Actor):
            >>> # Regression mode
            >>> model = ContinuousQAC(64, 6, 'regression')
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 6])
            >>> # Reparameterization Mode
            >>> model = ContinuousQAC(64, 6, 'reparameterization')
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['logit'][0].shape == torch.Size([4, 6])  # mu
            >>> actor_outputs['logit'][1].shape == torch.Size([4, 6]) # sigma

        Examples (Critic):
            >>> inputs = {'obs': torch.randn(4, 8), 'action': torch.randn(4, 1)}
            >>> model = ContinuousQAC(obs_shape=(8, ),action_shape=1, action_space='regression')
            >>> assert model(inputs, mode='compute_critic')['q_value'].shape == (4, )  # q value
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, obs: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Overview:
            QAC forward computation graph for actor part, input observation tensor to predict action or action logit.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]`): Actor output dict varying \
                from action_space: ``regression``, ``reparameterization``, ``hybrid``.
        ReturnsKeys (regression):
            - action (:obj:`torch.Tensor`): Continuous action with same size as ``action_shape``, usually in DDPG/TD3.
        ReturnsKeys (reparameterization):
            - logit (:obj:`Dict[str, torch.Tensor]`): The predictd reparameterization action logit, usually in SAC. \
                It is a list containing two tensors: ``mu`` and ``sigma``. The former is the mean of the gaussian \
                distribution, the latter is the standard deviation of the gaussian distribution.
        ReturnsKeys (hybrid):
            - logit (:obj:`torch.Tensor`): The predicted discrete action type logit, it will be the same dimension \
                as ``action_type_shape``, i.e., all the possible discrete action types.
            - action_args (:obj:`torch.Tensor`): Continuous action arguments with same size as ``action_args_shape``.
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
            >>> model = ContinuousQAC(64, 6, 'regression')
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 6])
            >>> # Reparameterization Mode
            >>> model = ContinuousQAC(64, 6, 'reparameterization')
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['logit'][0].shape == torch.Size([4, 6])  # mu
            >>> actor_outputs['logit'][1].shape == torch.Size([4, 6]) # sigma
        """
        obs = self.actor_encoder(obs)
        if self.action_space == 'regression':
            x = self.actor_head(obs)
            return {'action': x['pred']}
        elif self.action_space == 'reparameterization':
            x = self.actor_head(obs)
            return {'logit': [x['mu'], x['sigma']]}
        elif self.action_space == 'hybrid':
            logit = self.actor_head[0](obs)
            action_args = self.actor_head[1](obs)
            return {'logit': logit['logit'], 'action_args': action_args['pred']}

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            QAC forward computation graph for critic part, input observation and action tensor to predict Q-value.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The dict of input data, including ``obs`` and ``action`` \
                tensor, also contains ``logit`` and ``action_args`` tensor in hybrid action_space.
        ArgumentsKeys:
            - obs: (:obj:`torch.Tensor`): Observation tensor data, now supports a batch of 1-dim vector data.
            - action (:obj:`Union[torch.Tensor, Dict]`): Continuous action with same size as ``action_shape``.
            - logit (:obj:`torch.Tensor`): Discrete action logit, only in hybrid action_space.
            - action_args (:obj:`torch.Tensor`): Continuous action arguments, only in hybrid action_space.
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): The output dict of QAC's forward computation graph for critic, \
                including ``q_value``.
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
            >>> model = ContinuousQAC(obs_shape=(8, ),action_shape=1, action_space='regression')
            >>> assert model(inputs, mode='compute_critic')['q_value'].shape == (4, )  # q value
        """

        obs, action = inputs['obs'], inputs['action']
        obs = self.critic_encoder(obs)
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
            x = [m(x)['pred'] for m in self.critic_head]
        else:
            x = self.critic_head(x)['pred']
        return {'q_value': x}


@MODEL_REGISTRY.register('discrete_qac')
class DiscreteQAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to discrete action Q-value Actor-Critic (QAC), \
        such as DiscreteSAC. This model now supports only discrete action space. The DiscreteQAC is composed of \
        four parts: ``actor_encoder``, ``critic_encoder``, ``actor_head`` and ``critic_head``. Encoders are used to \
        extract the feature from various observation. Heads are used to predict corresponding Q-value or action logit. \
        In high-dimensional observation space like 2D image, we often use a shared encoder for both ``actor_encoder`` \
        and ``critic_encoder``. In low-dimensional observation space like 1D vector, we often use different encoders.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        twin_critic: bool = False,
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        encoder_hidden_size_list: SequenceType = None,
        share_encoder: Optional[bool] = False,
    ) -> None:
        """
        Overview:
            Initailize the DiscreteQAC Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's shape, such as 128, (156, ).
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's shape, such as 4, (3, ).
            - twin_critic (:obj:`bool`): Whether to use twin critic.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor head.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the actor network to compute action.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic head.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the critic network to compute Q-value.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` \
                after each FC layer, if ``None`` then default set to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization to after network layer (FC, Conv), \
                see ``ding.torch_utils.network`` for more details.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``, this argument is only used in image observation.
            - share_encoder (:obj:`Optional[bool]`): Whether to share encoder between actor and critic.
        """
        super(DiscreteQAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape: int = squeeze(action_shape)
        # encoder
        self.share_encoder = share_encoder
        if np.isscalar(obs_shape) or len(obs_shape) == 1:
            assert not self.share_encoder, "Vector observation doesn't need share encoder."
            assert encoder_hidden_size_list is None, "Vector obs encoder only uses one layer nn.Linear"
            # Because there is already a layer nn.Linear in the head, so we use nn.Identity here to keep
            # compatible with the image observation and avoid adding an extra layer nn.Linear.
            self.actor_encoder = nn.Identity()
            self.critic_encoder = nn.Identity()
            encoder_output_size = obs_shape
        elif len(obs_shape) == 3:

            def setup_conv_encoder():
                kernel_size = [3 for _ in range(len(encoder_hidden_size_list))]
                stride = [2] + [1 for _ in range(len(encoder_hidden_size_list) - 1)]
                return ConvEncoder(
                    obs_shape,
                    encoder_hidden_size_list,
                    activation=activation,
                    norm_type=norm_type,
                    kernel_size=kernel_size,
                    stride=stride
                )

            if self.share_encoder:
                encoder = setup_conv_encoder()
                self.actor_encoder = self.critic_encoder = encoder
            else:
                self.actor_encoder = setup_conv_encoder()
                self.critic_encoder = setup_conv_encoder()
            encoder_output_size = self.actor_encoder.output_size
        else:
            raise RuntimeError("not support observation shape: {}".format(obs_shape))

        # head
        self.actor_head = nn.Sequential(
            nn.Linear(encoder_output_size, actor_head_hidden_size), activation,
            DiscreteHead(
                actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type
            )
        )

        self.twin_critic = twin_critic
        if self.twin_critic:
            self.critic_head = nn.ModuleList()
            for _ in range(2):
                self.critic_head.append(
                    nn.Sequential(
                        nn.Linear(encoder_output_size, critic_head_hidden_size), activation,
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
            self.critic_head = nn.Sequential(
                nn.Linear(encoder_output_size, critic_head_hidden_size), activation,
                DiscreteHead(
                    critic_head_hidden_size,
                    action_shape,
                    critic_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        # Convenient for calling some apis (e.g. self.critic.parameters()),
        # but may cause misunderstanding when `print(self)`
        self.actor = nn.ModuleList([self.actor_encoder, self.actor_head])
        self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def forward(self, inputs: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
        """
        Overview:
            QAC forward computation graph, input observation tensor to predict Q-value or action logit. Different \
            ``mode`` will forward with different network modules to get different outputs and save computation.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The input observation tensor data.
            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of QAC forward computation graph, whose \
                key-values vary in different forward modes.
        Examples (Actor):
            >>> model = DiscreteQAC(64, 6)
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 6])

        Examples(Critic):
            >>> model = DiscreteQAC(64, 6, twin_critic=False)
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_critic')
            >>> assert actor_outputs['q_value'].shape == torch.Size([4, 6])
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Overview:
            QAC forward computation graph for actor part, input observation tensor to predict action or action logit.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): The output dict of QAC forward computation graph for actor, \
                including discrete action ``logit``.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted discrete action type logit, it will be the same dimension \
                as ``action_shape``, i.e., all the possible discrete action choices.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``obs_shape``.
            - logit (:obj:`torch.Tensor`): :math:`(B, N2)`, B is batch size and N2 corresponds to \
                ``action_shape``.
        Examples:
            >>> model = DiscreteQAC(64, 6)
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 6])
        """
        x = self.actor_encoder(inputs)
        x = self.actor_head(x)
        return {'logit': x['logit']}

    def compute_critic(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Overview:
            QAC forward computation graph for critic part, input observation to predict Q-value for each possible \
            discrete action choices.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): The output dict of QAC forward computation graph for critic, \
                including ``q_value`` for each possible discrete action choices.
        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): The predicted Q-value for each possible discrete action choices, it will \
                be the same dimension as ``action_shape`` and used to calculate the loss.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``.
            - q_value (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``.
        Examples:
            >>> model = DiscreteQAC(64, 6, twin_critic=False)
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_critic')
            >>> assert actor_outputs['q_value'].shape == torch.Size([4, 6])
        """
        inputs = self.critic_encoder(inputs)
        if self.twin_critic:
            x = [m(inputs)['logit'] for m in self.critic_head]
        else:
            x = self.critic_head(inputs)['logit']
        return {'q_value': x}

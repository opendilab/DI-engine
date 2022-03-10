from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


@MODEL_REGISTRY.register('maqac')
class MAQAC(nn.Module):
    r"""
    Overview:
        The MAQAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
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
            Init the MAQAC Model according to arguments.
        Arguments:
            - agent_obs_shape (:obj:`Union[int, SequenceType]`): Agent's observation's space.
            - global_obs_shape (:obj:`Union[int, SequenceType]`): Global observation's space.
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
        super(MAQAC, self).__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.actor = nn.Sequential(
            nn.Linear(agent_obs_shape, actor_head_hidden_size), activation,
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
                        nn.Linear(global_obs_shape, critic_head_hidden_size), activation,
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
                nn.Linear(global_obs_shape, critic_head_hidden_size), activation,
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
            Use observation and action tensor to predict output.
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
                    - logit (:obj:`torch.Tensor`): Action's probabilities.
                Forward with ``'compute_critic'``, Necessary Keys:
                    - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Actor Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.
        Critic Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``global_obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: Dict) -> Dict:
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
        action_mask = inputs['obs']['action_mask']
        x = self.actor(inputs['obs']['agent_state'])
        return {'logit': x['logit'], 'action_mask': action_mask}

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
        """

        if self.twin_critic:
            x = [m(inputs['obs']['global_state'])['logit'] for m in self.critic]
        else:
            x = self.critic(inputs['obs']['global_state'])['logit']
        return {'q_value': x}


@MODEL_REGISTRY.register('maqac_continuous')
class ContinuousMAQAC(nn.Module):
    r"""
    Overview:
        The Continuous MAQAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
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
        r"""
        Overview:
            Init the QAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's space, such as 4, (3, )
            - action_space (:obj:`str`): Whether choose ``regression`` or ``reparameterization``.
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
        super(ContinuousMAQAC, self).__init__()
        obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization']
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
        else:  # SAC
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
        self.twin_critic = twin_critic
        critic_input_size = global_obs_shape + action_shape
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

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use observation and action tensor to predict output.
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

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: Dict) -> Dict:
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
            - logit + action_args
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - logit (:obj:`Union[list, torch.Tensor]`):
              - case1(continuous space, list): 2 elements, mu and sigma, each is the shape of :math:`(B, N0)`.
              - case2(hybrid space, torch.Tensor): :math:`(B, N1)`, where N1 is action_type_shape
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, B is batch size.
            - action_args (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where N2 is action_args_shape
                (action_args are continuous real value)
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
        inputs = inputs['agent_state']
        if self.action_space == 'regression':
            x = self.actor(inputs)
            return {'action': x['pred']}
        else:
            x = self.actor(inputs)
            return {'logit': [x['mu'], x['sigma']]}

    def compute_critic(self, inputs: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj: `Dict`): ``obs``, ``action`` and ``logit` tensors.
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Q-value output.

        ArgumentsKeys:
            - necessary:
              - obs: (:obj:`torch.Tensor`): 2-dim vector observation
              - action (:obj:`Union[torch.Tensor, Dict]`): action from actor
            - optional:
              - logit (:obj:`torch.Tensor`): discrete action logit
        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.
        """

        obs, action = inputs['obs']['global_state'], inputs['action']
        if len(action.shape) == 1:  # (B, ) -> (B, 1)
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=-1)
        if self.twin_critic:
            x = [m(x)['pred'] for m in self.critic]
        else:
            x = self.critic(x)['pred']
        return {'q_value': x}

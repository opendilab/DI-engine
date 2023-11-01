from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


@MODEL_REGISTRY.register('discrete_maqac')
class DiscreteMAQAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to discrete action Multi-Agent Q-value \
        Actor-CritiC (MAQAC) model. The model is composed of actor and critic, where actor is a MLP network and \
        critic is a MLP network. The actor network is used to predict the action probability distribution, and the \
        critic network is used to predict the Q value of the state-action pair.
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
        """
        Overview:
            Initialize the DiscreteMAQAC Model according to arguments.
        Arguments:
            - agent_obs_shape (:obj:`Union[int, SequenceType]`): Agent's observation's space.
            - global_obs_shape (:obj:`Union[int, SequenceType]`): Global observation's space.
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - twin_critic (:obj:`bool`): Whether include twin critic.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for critic's nn.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` the after \
                ``layer_fn``, if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization to use, see ``ding.torch_utils.fc_block`` \
                for more details.
        """
        super(DiscreteMAQAC, self).__init__()
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
        """
        Overview:
            Use observation tensor to predict output, with ``'compute_actor'`` or ``'compute_critic'`` mode.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): The agent's observation tensor data, \
                        with shape :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): The global observation tensor data, \
                        with shape :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): The action mask tensor data, \
                        with shape :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of DiscreteMAQAC forward computation graph, whose \
                key-values vary in different forward modes.
            Forward with ``'compute_actor'``, Necessary Keys (either):
                - logit (:obj:`torch.Tensor`): Action's probabilities.
                - action_mask (:obj:`torch.Tensor`): Action mask tensor with same size as ``action_shape``.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): Q value tensor is the shape of :math:`(B, A, N2)`, where B is batch size \
                    and A is agent num. N2 corresponds to ``action_shape``.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N2)`, where B is batch size and \
                    A is agent num. N2 corresponds to ``action_shape``.
        Shapes:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of DiscreteMAQAC forward computation graph, whose \
                key-values vary in different forward modes.
            Forward with ``'compute_actor'``, Necessary Keys (either):
                - logit (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                    N2 corresponds to ``action_shape``.
                - action_mask (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                    N2 corresponds to ``action_shape``.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N2)`, where B is batch size and \
                    A is agent num. N2 corresponds to ``action_shape``.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                    N2 corresponds to ``action_shape``.
        Examples:
            >>> B = 32
            >>> agent_obs_shape = 216
            >>> global_obs_shape = 264
            >>> agent_num = 8
            >>> action_shape = 14
            >>> data = {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(B, agent_num, agent_obs_shape),
            >>>         'global_state': torch.randn(B, agent_num, global_obs_shape),
            >>>         'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            >>>     }
            >>> }
            >>> model = DiscreteMAQAC(agent_obs_shape, global_obs_shape, action_shape, twin_critic=True)
            >>> logit = model(data, mode='compute_actor')['logit']
            >>> value = model(data, mode='compute_critic')['q_value']
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: Dict) -> Dict:
        """
        Overview:
            Use observation tensor to predict action logits.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): The agent's observation tensor data, \
                        with shape :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): The global observation tensor data, \
                        with shape :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): The action mask tensor data, \
                        with shape :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of DiscreteMAQAC forward computation graph, whose \
                key-values vary in different forward modes.
                - logit (:obj:`torch.Tensor`): Action's probabilities.
                - action_mask (:obj:`torch.Tensor`): Action mask tensor with same size as ``action_shape``.
        Shapes:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of DiscreteMAQAC forward computation graph, whose \
                key-values vary in different forward modes.
                - logit (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                    N2 corresponds to ``action_shape``.
                - action_mask (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                    N2 corresponds to ``action_shape``.
        Examples:
            >>> B = 32
            >>> agent_obs_shape = 216
            >>> global_obs_shape = 264
            >>> agent_num = 8
            >>> action_shape = 14
            >>> data = {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(B, agent_num, agent_obs_shape),
            >>>         'global_state': torch.randn(B, agent_num, global_obs_shape),
            >>>         'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            >>>     }
            >>> }
            >>> model = DiscreteMAQAC(agent_obs_shape, global_obs_shape, action_shape, twin_critic=True)
            >>> logit = model.compute_actor(data)['logit']
        """
        action_mask = inputs['obs']['action_mask']
        x = self.actor(inputs['obs']['agent_state'])
        return {'logit': x['logit'], 'action_mask': action_mask}

    def compute_critic(self, inputs: Dict) -> Dict:
        """
        Overview:
            use observation tensor to predict Q value.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): The agent's observation tensor data, \
                        with shape :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): The global observation tensor data, \
                        with shape :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): The action mask tensor data, \
                        with shape :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of DiscreteMAQAC forward computation graph, whose \
                key-values vary in different forward modes.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): Q value tensor is the shape of :math:`(B, A, N2)`, where B is batch size \
                    and A is agent num. N2 corresponds to ``action_shape``.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N2)`, where B is batch size and \
                    A is agent num. N2 corresponds to ``action_shape``.
        Shapes:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
            - output (:obj:`Dict[str, torch.Tensor]`): The output dict of DiscreteMAQAC forward computation graph, whose \
                key-values vary in different forward modes.
            if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N2)`, where B is batch size and \
                    A is agent num. N2 corresponds to ``action_shape``.
            if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                    N2 corresponds to ``action_shape``.
        Examples:
            >>> B = 32
            >>> agent_obs_shape = 216
            >>> global_obs_shape = 264
            >>> agent_num = 8
            >>> action_shape = 14
            >>> data = {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(B, agent_num, agent_obs_shape),
            >>>         'global_state': torch.randn(B, agent_num, global_obs_shape),
            >>>         'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            >>>     }
            >>> }
            >>> model = DiscreteMAQAC(agent_obs_shape, global_obs_shape, action_shape, twin_critic=True)
            >>> value = model.compute_critic(data)['q_value']
        """

        if self.twin_critic:
            x = [m(inputs['obs']['global_state'])['logit'] for m in self.critic]
        else:
            x = self.critic(inputs['obs']['global_state'])['logit']
        return {'q_value': x}


@MODEL_REGISTRY.register('continuous_maqac')
class ContinuousMAQAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to continuous action Multi-Agent Q-value \
        Actor-CritiC (MAQAC) model. The model is composed of actor and critic, where actor is a MLP network and \
        critic is a MLP network. The actor network is used to predict the action probability distribution, and the \
        critic network is used to predict the Q value of the state-action pair.
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
        """
        Overview:
            Initialize the QAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's space, such as 4, (3, )
            - action_space (:obj:`str`): Whether choose ``regression`` or ``reparameterization``.
            - twin_critic (:obj:`bool`): Whether include twin critic.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for critic's nn.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` the after \
                ``layer_fn``, if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization to use, see ``ding.torch_utils.fc_block`` \
                for more details.
        """
        super(ContinuousMAQAC, self).__init__()
        obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization'], self.action_space
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
        """
        Overview:
            Use observation and action tensor to predict output in ``'compute_actor'`` or ``'compute_critic'`` mode.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): The agent's observation tensor data, \
                        with shape :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): The global observation tensor data, \
                        with shape :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): The action mask tensor data, \
                        with shape :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
                - ``action`` (:obj:`torch.Tensor`): The action tensor data, \
                    with shape :math:`(B, A, N3)`, where B is batch size and A is agent num. \
                    N3 corresponds to ``action_shape``.
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.
            Forward with ``'compute_actor'``, if action_space == 'regression', Necessary Keys:
                - action (:obj:`torch.Tensor`): Action tensor with same size as ``action_shape``.
            Forward with ``'compute_actor'``, if action_space == 'reparameterization', Necessary Keys:
                - logit (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N3)`, where B is batch size and \
                    A is agent num. N3 corresponds to ``action_shape``.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A)`, where B is batch size and \
                    A is agent num.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is agent num.
        Shapes:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
                - ``action`` (:obj:`torch.Tensor`): :math:`(B, A, N3)`, where B is batch size and A is agent num. \
                    N3 corresponds to ``action_shape``.
            - outputs (:obj:`Dict`): Outputs of network forward.
            Forward with ``'compute_actor'``, if action_space == 'regression', Necessary Keys:
                - action (:obj:`torch.Tensor`): :math:`(B, A, N3)`, where B is batch size and A is agent num. \
                    N3 corresponds to ``action_shape``.
            Forward with ``'compute_actor'``, if action_space == 'reparameterization', Necessary Keys:
                - logit (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N3)`, where B is batch size and \
                    A is agent num. N3 corresponds to ``action_shape``.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A)`, where B is batch size and \
                    A is agent num.
            Forward with ``'compute_critic'``, if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is agent num.
        Examples:
            >>> B = 32
            >>> agent_obs_shape = 216
            >>> global_obs_shape = 264
            >>> agent_num = 8
            >>> action_shape = 14
            >>> action_space = 'regression'
            >>> # or
            >>> action_space = 'reparameterization'
            >>> data = {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(B, agent_num, agent_obs_shape),
            >>>         'global_state': torch.randn(B, agent_num, global_obs_shape),
            >>>         'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            >>>     },
            >>>     'action': torch.randn(B, agent_num, squeeze(action_shape))
            >>> }
            >>> model = ContinuousMAQAC(agent_obs_shape, global_obs_shape, action_shape, action_space, twin_critic=False)
            >>> if action_space == 'regression':
            >>>     action = model(data['obs'], mode='compute_actor')['action']
            >>> elif action_space == 'reparameterization':
            >>>     (mu, sigma) = model(data['obs'], mode='compute_actor')['logit']
            >>> value = model(data, mode='compute_critic')['q_value']
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: Dict) -> Dict:
        """
        Overview:
            Use observation tensor to predict action logits.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``agent_state`` (:obj:`torch.Tensor`): The agent's observation tensor data, \
                    with shape :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                    N0 corresponds to ``agent_obs_shape``.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.
            if action_space == 'regression', Necessary Keys:
                - action (:obj:`torch.Tensor`): Action tensor with same size as ``action_shape``.
            if action_space == 'reparameterization', Necessary Keys:
                - logit (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N3)`, where B is batch size and \
                    A is agent num. N3 corresponds to ``action_shape``.
        Shapes:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``agent_state`` (:obj:`torch.Tensor`): :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                    N0 corresponds to ``agent_obs_shape``.
            - outputs (:obj:`Dict`): Outputs of network forward.
            if action_space == 'regression', Necessary Keys:
                - action (:obj:`torch.Tensor`): :math:`(B, A, N3)`, where B is batch size and A is agent num. \
                    N3 corresponds to ``action_shape``.
            if action_space == 'reparameterization', Necessary Keys:
                - logit (:obj:`list`): 2 elements, each is the shape of :math:`(B, A, N3)`, where B is batch size and \
                    A is agent num. N3 corresponds to ``action_shape``.
        Examples:
            >>> B = 32
            >>> agent_obs_shape = 216
            >>> global_obs_shape = 264
            >>> agent_num = 8
            >>> action_shape = 14
            >>> action_space = 'regression'
            >>> # or
            >>> action_space = 'reparameterization'
            >>> data = {
            >>>     'agent_state': torch.randn(B, agent_num, agent_obs_shape),
            >>> }
            >>> model = ContinuousMAQAC(agent_obs_shape, global_obs_shape, action_shape, action_space, twin_critic=False)
            >>> if action_space == 'regression':
            >>>     action = model.compute_actor(data)['action']
            >>> elif action_space == 'reparameterization':
            >>>     (mu, sigma) = model.compute_actor(data)['logit']
        """
        inputs = inputs['agent_state']
        if self.action_space == 'regression':
            x = self.actor(inputs)
            return {'action': x['pred']}
        else:
            x = self.actor(inputs)
            return {'logit': [x['mu'], x['sigma']]}

    def compute_critic(self, inputs: Dict) -> Dict:
        """
        Overview:
            Use observation tensor and action tensor to predict Q value.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): The agent's observation tensor data, \
                        with shape :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): The global observation tensor data, \
                        with shape :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): The action mask tensor data, \
                        with shape :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
                - ``action`` (:obj:`torch.Tensor`): The action tensor data, \
                    with shape :math:`(B, A, N3)`, where B is batch size and A is agent num. \
                    N3 corresponds to ``action_shape``.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.
            if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A)`, where B is batch size and \
                    A is agent num.
            if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is agent num.
        Shapes:
            - inputs (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                - ``obs`` (:obj:`Dict[str, torch.Tensor]`): The input dict tensor data, has keys:
                    - ``agent_state`` (:obj:`torch.Tensor`): :math:`(B, A, N0)`, where B is batch size and A is agent num. \
                        N0 corresponds to ``agent_obs_shape``.
                    - ``global_state`` (:obj:`torch.Tensor`): :math:`(B, A, N1)`, where B is batch size and A is agent num. \
                        N1 corresponds to ``global_obs_shape``.
                    - ``action_mask`` (:obj:`torch.Tensor`): :math:`(B, A, N2)`, where B is batch size and A is agent num. \
                        N2 corresponds to ``action_shape``.
                - ``action`` (:obj:`torch.Tensor`): :math:`(B, A, N3)`, where B is batch size and A is agent num. \
                    N3 corresponds to ``action_shape``.
            - outputs (:obj:`Dict`): Outputs of network forward.
            if ``twin_critic`` is ``True``, Necessary Keys:
                - q_value (:obj:`list`): 2 elements, each is the shape of :math:`(B, A)`, where B is batch size and \
                    A is agent num.
            if ``twin_critic`` is ``False``, Necessary Keys:
                - q_value (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is agent num.
        Examples:
            >>> B = 32
            >>> agent_obs_shape = 216
            >>> global_obs_shape = 264
            >>> agent_num = 8
            >>> action_shape = 14
            >>> action_space = 'regression'
            >>> # or
            >>> action_space = 'reparameterization'
            >>> data = {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(B, agent_num, agent_obs_shape),
            >>>         'global_state': torch.randn(B, agent_num, global_obs_shape),
            >>>         'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            >>>     },
            >>>     'action': torch.randn(B, agent_num, squeeze(action_shape))
            >>> }
            >>> model = ContinuousMAQAC(agent_obs_shape, global_obs_shape, action_shape, action_space, twin_critic=False)
            >>> value = model.compute_critic(data)['q_value']
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

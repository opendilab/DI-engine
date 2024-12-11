from typing import Union, Dict, Tuple, Optional
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead


@MODEL_REGISTRY.register('mavac')
class MAVAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to (state) Value Actor-Critic (VAC) for \
        multi-agent, such as MAPPO(https://arxiv.org/abs/2103.01955). This model now supports discrete and \
        continuous action space. The MAVAC is composed of four parts: ``actor_encoder``, ``critic_encoder``, \
        ``actor_head`` and ``critic_head``. Encoders are used to extract the feature from various observation. \
        Heads are used to predict corresponding value or action logit.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``, ``compute_actor_critic``.
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        agent_obs_shape: Union[int, SequenceType],
        global_obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        agent_num: int,
        actor_head_hidden_size: int = 256,
        actor_head_layer_num: int = 2,
        critic_head_hidden_size: int = 512,
        critic_head_layer_num: int = 1,
        action_space: str = 'discrete',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        bound_type: Optional[str] = None,
        encoder: Optional[Tuple[torch.nn.Module, torch.nn.Module]] = None,
    ) -> None:
        """
        Overview:
            Init the MAVAC Model according to arguments.
        Arguments:
            - agent_obs_shape (:obj:`Union[int, SequenceType]`): Observation's space for single agent, \
                such as 8 or [4, 84, 84].
            - global_obs_shape (:obj:`Union[int, SequenceType]`): Global observation's space, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape for single agent, such as 6 \
                or [2, 3, 3].
            - agent_num (:obj:`int`): This parameter is temporarily reserved. This parameter may be required for \
                subsequent changes to the model
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``actor_head`` network, defaults \
                to 256, it must match the last element of ``agent_obs_shape``.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the ``actor_head`` network to compute action.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``critic_head`` network, defaults \
                to 512, it must match the last element of ``global_obs_shape``.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output for \
                critic's nn.
            - action_space (:obj:`Union[int, SequenceType]`): The type of different action spaces, including \
                ['discrete', 'continuous'], then will instantiate corresponding head, including ``DiscreteHead`` \
                and ``ReparameterizationHead``.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` the after \
                ``layer_fn``, if ``None`` then default set to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN'].
            - sigma_type (:obj:`Optional[str]`): The type of sigma in continuous action space, see \
                ``ding.torch_utils.network.dreamer.ReparameterizationHead`` for more details, in MAPPO, it defaults \
                to ``independent``, which means state-independent sigma parameters.
            - bound_type (:obj:`Optional[str]`): The type of action bound methods in continuous action space, defaults \
                to ``None``, which means no bound.
            - encoder (:obj:`Optional[Tuple[torch.nn.Module, torch.nn.Module]]`): The encoder module list, defaults \
                to ``None``, you can define your own actor and critic encoder module and pass it into MAVAC to \
                deal with different observation space.
        """
        super(MAVAC, self).__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape, self.agent_obs_shape, self.action_shape = global_obs_shape, agent_obs_shape, action_shape
        self.action_space = action_space
        # Encoder Type
        if encoder:
            self.actor_encoder, self.critic_encoder = encoder
        else:
            # We directly connect the Head after a Liner layer instead of using the 3-layer FCEncoder.
            # In SMAC task it can obviously improve the performance.
            # Users can change the model according to their own needs.
            self.actor_encoder = nn.Sequential(
                nn.Linear(agent_obs_shape, actor_head_hidden_size),
                activation,
            )
            self.critic_encoder = nn.Sequential(
                nn.Linear(global_obs_shape, critic_head_hidden_size),
                activation,
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
        self.actor = [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_head]
        # for convenience of call some apis(such as: self.critic.parameters()), but may cause
        # misunderstanding when print(self)
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        """
        Overview:
            MAVAC forward computation graph, input observation tensor to predict state value or action logit. \
            ``mode`` includes ``compute_actor``, ``compute_critic``, ``compute_actor_critic``.
            Different ``mode`` will forward with different network modules to get different outputs and save \
            computation.
        Arguments:
            - inputs (:obj:`Dict`): The input dict including observation and related info, \
                whose key-values vary from different ``mode``.
            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.
        Returns:
            - outputs (:obj:`Dict`): The output dict of MAVAC's forward computation graph, whose key-values vary from \
                different ``mode``.

        Examples (Actor):
            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)
            >>> inputs = {
                    'agent_state': torch.randn(10, 8, 64),
                    'global_state': torch.randn(10, 8, 128),
                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))
                }
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([10, 8, 14])

        Examples (Critic):
            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)
            >>> inputs = {
                    'agent_state': torch.randn(10, 8, 64),
                    'global_state': torch.randn(10, 8, 128),
                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))
                }
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> assert actor_outputs['value'].shape == torch.Size([10, 8])

        Examples (Actor-Critic):
            >>> model = MAVAC(64, 64)
            >>> inputs = {
                    'agent_state': torch.randn(10, 8, 64),
                    'global_state': torch.randn(10, 8, 128),
                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))
                }
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> assert outputs['value'].shape == torch.Size([10, 8, 14])
            >>> assert outputs['logit'].shape == torch.Size([10, 8])

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: Dict) -> Dict:
        """
        Overview:
            MAVAC forward computation graph for actor part, \
            predicting action logit with agent observation tensor in ``x``.
        Arguments:
            - x (:obj:`Dict`): Input data dict with keys ['agent_state', 'action_mask'(optional)].
                - agent_state: (:obj:`torch.Tensor`): Each agent local state(obs).
                - action_mask(optional): (:obj:`torch.Tensor`): When ``action_space`` is discrete, action_mask needs \
                    to be provided to mask illegal actions.
        Returns:
            - outputs (:obj:`Dict`): The output dict of the forward computation graph for actor, including ``logit``.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be \
                the same dimension real-value ranged tensor of possible action choices, and for continuous action \
                space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the \
                same as the number of continuous actions.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M, N)`, where B is batch size and N is ``action_shape`` \
              and M is ``agent_num``.

        Examples:
            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)
            >>> inputs = {
                    'agent_state': torch.randn(10, 8, 64),
                    'global_state': torch.randn(10, 8, 128),
                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))
                }
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([10, 8, 14])

        """
        if self.action_space == 'discrete':
            action_mask = x['action_mask']
            x = x['agent_state']
            x = self.actor_encoder(x)
            x = self.actor_head(x)
            logit = x['logit']
            logit[action_mask == 0.0] = -99999999
        elif self.action_space == 'continuous':
            x = x['agent_state']
            x = self.actor_encoder(x)
            x = self.actor_head(x)
            logit = x
        return {'logit': logit}

    def compute_critic(self, x: Dict) -> Dict:
        """
        Overview:
            MAVAC forward computation graph for critic part. \
            Predict state value with global observation tensor in ``x``.
        Arguments:
            - x (:obj:`Dict`): Input data dict with keys ['global_state'].
                - global_state: (:obj:`torch.Tensor`): Global state(obs).
        Returns:
            - outputs (:obj:`Dict`): The output dict of MAVAC's forward computation graph for critic, \
                including ``value``.
        ReturnsKeys:
            - value (:obj:`torch.Tensor`): The predicted state value tensor.
        Shapes:
            - value (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is ``agent_num``.

        Examples:
            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)
            >>> inputs = {
                    'agent_state': torch.randn(10, 8, 64),
                    'global_state': torch.randn(10, 8, 128),
                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))
                }
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> assert critic_outputs['value'].shape == torch.Size([10, 8])
        """

        x = self.critic_encoder(x['global_state'])
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: Dict) -> Dict:
        """
        Overview:
            MAVAC forward computation graph for both actor and critic part, input observation to predict action \
            logit and state value.
        Arguments:
            - x (:obj:`Dict`): The input dict contains ``agent_state``, ``global_state`` and other related info.
        Returns:
            - outputs (:obj:`Dict`): The output dict of MAVAC's forward computation graph for both actor and critic, \
                including ``logit`` and ``value``.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.
            - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M, N)`, where B is batch size and N is ``action_shape`` \
                and M is ``agent_num``.
            - value (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch sizeand M is ``agent_num``.

        Examples:
            >>> model = MAVAC(64, 64)
            >>> inputs = {
                    'agent_state': torch.randn(10, 8, 64),
                    'global_state': torch.randn(10, 8, 128),
                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))
                }
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> assert outputs['value'].shape == torch.Size([10, 8])
            >>> assert outputs['logit'].shape == torch.Size([10, 8, 14])
        """
        x_actor = self.actor_encoder(x['agent_state'])
        x_critic = self.critic_encoder(x['global_state'])

        if self.action_space == 'discrete':
            action_mask = x['action_mask']
            x = self.actor_head(x_actor)
            logit = x['logit']
            logit[action_mask == 0.0] = -99999999
        elif self.action_space == 'continuous':
            x = self.actor_head(x_actor)
            logit = x
        value = self.critic_head(x_critic)['pred']
        return {'logit': logit, 'value': value}

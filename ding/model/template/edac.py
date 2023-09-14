from typing import Union, Optional, Dict
from easydict import EasyDict

import torch
import torch.nn as nn
from ding.model.common import ReparameterizationHead, EnsembleHead
from ding.utils import SequenceType, squeeze

from ding.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register('edac')
class EDAC(nn.Module):
    """
    Overview:
        The Q-value Actor-Critic network with the ensemble mechanism, which is used in EDAC.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType, EasyDict],
            ensemble_num: int = 2,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Overview:
            Initailize the EDAC Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's shape, such as 128, (156, ).
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's shape, such as 4, (3, ), \
                EasyDict({'action_type_shape': 3, 'action_args_shape': 4}).
            - ensemble_num (:obj:`int`): Q-net number.
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
        super(EDAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.ensemble_num = ensemble_num
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

        critic_input_size = obs_shape + action_shape
        self.critic = EnsembleHead(
            critic_input_size,
            1,
            critic_head_hidden_size,
            critic_head_layer_num,
            self.ensemble_num,
            activation=activation,
            norm_type=norm_type
        )

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], mode: str) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The unique execution (forward) method of EDAC method, and one can indicate different modes to implement \
            different computation graph, including ``compute_actor`` and ``compute_critic`` in EDAC.
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
                from action_space: ``reparameterization``.
        ReturnsKeys (either):
                - logit (:obj:`Dict[str, torch.Tensor]`): Reparameterization logit, usually in SAC.
                    - mu (:obj:`torch.Tensor`): Mean of parameterization gaussion distribution.
                    - sigma (:obj:`torch.Tensor`): Standard variation of parameterization gaussion distribution.
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
            >>> model = EDAC(64, 64,)
            >>> obs = torch.randn(4, 64)
            >>> actor_outputs = model(obs,'compute_actor')
            >>> assert actor_outputs['logit'][0].shape == torch.Size([4, 64])  # mu
            >>> actor_outputs['logit'][1].shape == torch.Size([4, 64]) # sigma
        """
        x = self.actor(obs)
        return {'logit': [x['mu'], x['sigma']]}

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The forward computation graph of compute_critic mode, uses observation and action tensor to produce critic
            output, such as ``q_value``.
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): Dict strcture of input data, including ``obs`` and \
                  ``action`` tensor
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): Critic output, such as ``q_value``.
        ArgumentsKeys:
            - obs: (:obj:`torch.Tensor`): Observation tensor data, now supports a batch of 1-dim vector data.
            - action (:obj:`Union[torch.Tensor, Dict]`): Continuous action with same size as ``action_shape``.
        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)` or '(Ensemble_num, B, N1)', where B is batch size and N1 is \
                  ``obs_shape``.
            - action (:obj:`torch.Tensor`): :math:`(B, N2)` or '(Ensemble_num, B, N2)', where B is batch size and N4 \
                  is ``action_shape``.
            - q_value (:obj:`torch.Tensor`): :math:`(Ensemble_num, B)`, where B is batch size.
        Examples:
            >>> inputs = {'obs': torch.randn(4, 8), 'action': torch.randn(4, 1)}
            >>> model = EDAC(obs_shape=(8, ),action_shape=1)
            >>> model(inputs, mode='compute_critic')['q_value']  # q value
            ... tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)
        """

        obs, action = inputs['obs'], inputs['action']
        if len(action.shape) == 1:  # (B, ) -> (B, 1)
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=-1)
        if len(obs.shape) < 3:
            # [batch_size,dim] -> [batch_size,Ensemble_num * dim,1]
            x = x.repeat(1, self.ensemble_num).unsqueeze(-1)
        else:
            # [Ensemble_num,batch_size,dim] -> [batch_size,Ensemble_num,dim] -> [batch_size,Ensemble_num * dim, 1]
            x = x.transpose(0, 1)
            batch_size = obs.shape[1]
            x = x.reshape(batch_size, -1, 1)
        # [Ensemble_num,batch_size,1]
        x = self.critic(x)['pred']
        # [batch_size,1*Ensemble_num] -> [Ensemble_num,batch_size]
        x = x.permute(1, 0)
        return {'q_value': x}

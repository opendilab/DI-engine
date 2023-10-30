from typing import Optional, Dict, Union
import copy
import torch
import torch.nn as nn
from ding.utils import SequenceType, MODEL_REGISTRY
from .vac import VAC


@MODEL_REGISTRY.register('ppg')
class PPG(nn.Module):
    """
    Overview:
        Phasic Policy Gradient (PPG) model from paper `Phasic Policy Gradient`
        https://arxiv.org/abs/2009.04416 \
        This module contains VAC module and an auxiliary critic module.
    Interfaces:
        ``forward``, ``compute_actor``, ``compute_critic``, ``compute_actor_critic``
    """

    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        action_space: str = 'discrete',
        share_encoder: bool = True,
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 2,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        impala_cnn_encoder: bool = False,
    ) -> None:
        """
        Overview:
            Initailize the PPG Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's shape, such as 128, (156, ).
            - action_shape (:obj:`Union[int, SequenceType]`): Action's shape, such as 4, (3, ).
            - action_space (:obj:`str`): The action space type, such as 'discrete', 'continuous'.
            - share_encoder (:obj:`bool`): Whether to share encoder.
            - encoder_hidden_size_list (:obj:`SequenceType`): The hidden size list of encoder.
            - actor_head_hidden_size (:obj:`int`): The ``hidden_size`` to pass to actor head.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for actor head.
            - critic_head_hidden_size (:obj:`int`): The ``hidden_size`` to pass to critic head.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for critic head.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` \
                after each FC layer, if ``None`` then default set to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization to after network layer (FC, Conv), \
                see ``ding.torch_utils.network`` for more details.
            - impala_cnn_encoder (:obj:`bool`): Whether to use impala cnn encoder.
        """
        super(PPG, self).__init__()
        self.actor_critic = VAC(
            obs_shape,
            action_shape,
            action_space,
            share_encoder,
            encoder_hidden_size_list,
            actor_head_hidden_size,
            actor_head_layer_num,
            critic_head_hidden_size,
            critic_head_layer_num,
            activation,
            norm_type,
            impala_cnn_encoder=impala_cnn_encoder
        )
        self.aux_critic = copy.deepcopy(self.actor_critic.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        """
        Overview:
            Compute action logits or value according to mode being ``compute_actor``, ``compute_critic`` or \
                ``compute_actor_critic``.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.
        Returns:
            - outputs (:obj:`Dict`): The output dict of PPG's forward computation graph, whose key-values vary from \
                different ``mode``.
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            Use actor to compute action logits.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - output (:obj:`Dict`): The output data containing action logits.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be \
                the same dimension real-value ranged tensor of possible action choices, and for continuous action \
                space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the \
                same as the number of continuous actions. Hybrid action space is a kind of combination of discrete \
                and continuous action space, so the logit will be a dict with ``action_type`` and ``action_args``.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is the input feature size.
            - output (:obj:`Dict`): ``logit``: :math:`(B, A)`, where B is batch size and A is the action space size.
        """
        return self.actor_critic(x, mode='compute_actor')

    def compute_critic(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            Use critic to compute value.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - output (:obj:`Dict`): The output dict of VAC's forward computation graph for critic, including ``value``.
        ReturnsKeys:
            - necessary: ``value``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is the input feature size.
            - output (:obj:`Dict`): ``value``: :math:`(B, 1)`, where B is batch size.
        """
        x = self.aux_critic[0](x)  # encoder
        x = self.aux_critic[1](x)  # head
        return {'value': x['pred']}

    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            Use actor and critic to compute action logits and value.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict`): The output dict of PPG's forward computation graph for both actor and critic, \
                including ``logit`` and ``value``.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be \
                the same dimension real-value ranged tensor of possible action choices, and for continuous action \
                space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the \
                same as the number of continuous actions. Hybrid action space is a kind of combination of discrete \
                and continuous action space, so the logit will be a dict with ``action_type`` and ``action_args``.
            - value (:obj:`torch.Tensor`): The predicted state value tensor.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is the input feature size.
            - output (:obj:`Dict`): ``value``: :math:`(B, 1)`, where B is batch size.
            - output (:obj:`Dict`): ``logit``: :math:`(B, A)`, where B is batch size and A is the action space size.

        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder.
        """
        return self.actor_critic(x, mode='compute_actor_critic')

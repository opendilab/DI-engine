from typing import Optional, Dict, Union
import copy
import torch
import torch.nn as nn
from ding.utils import SequenceType, MODEL_REGISTRY
from .vac import VAC


@MODEL_REGISTRY.register('ppg')
class PPG(nn.Module):
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
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``
        """
        return self.actor_critic(x, mode='compute_actor')

    def compute_critic(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``value``
        """
        x = self.aux_critic[0](x)  # encoder
        x = self.aux_critic[1](x)  # head
        return {'value': x['pred']}

    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        """
        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder
        ReturnsKeys:
            - necessary: ``value``, ``logit``
        """
        return self.actor_critic(x, mode='compute_actor_critic')

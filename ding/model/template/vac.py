from typing import Union, Dict, Optional
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


@MODEL_REGISTRY.register('vac')
class VAC(nn.Module):
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            share_encoder: bool = True,
            continuous: bool = False,
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 2,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        super(VAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape: int = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        # Encoder Type
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.encoder = encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            self.actor_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
            self.critic_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
        # Head Type
        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )
        self.continuous = continuous
        if self.continuous:
            self.multi_head = False
            self.actor_head = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type='independent',
                activation=activation,
                norm_type=norm_type
            )
        else:
            actor_head_cls = DiscreteHead
            multi_head = not isinstance(action_shape, int)
            self.multi_head = multi_head
            if multi_head:
                self.actor_head = MultiHead(
                    actor_head_cls,
                    actor_head_hidden_size,
                    action_shape,
                    layer_num=actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            else:
                self.actor_head = actor_head_cls(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
        # must use list, not nn.ModuleList
        if self.share_encoder:
            self.actor = [self.encoder, self.actor_head]
            self.critic = [self.encoder, self.critic_head]
        else:
            self.actor = [self.actor_encoder, self.actor_head]
            self.critic = [self.critic_encoder, self.critic_head]
        # for convenience of call some apis(such as: self.critic.parameters()), but may cause
        # misunderstanding when print(self)
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``logit``
        """
        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.actor_encoder(x)
        x = self.actor_head(x)
        if self.continuous:
            x = {'logit': [x['mu'], x['sigma']]}
        return x

    def compute_critic(self, x: torch.Tensor) -> Dict:
        """
        ReturnsKeys:
            - necessary: ``value``
        """
        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.critic_encoder(x)
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        """
        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder
        ReturnsKeys:
            - necessary: ``value``, ``logit``
        """
        if self.share_encoder:
            actor_embedding = critic_embedding = self.encoder(x)
        else:
            actor_embedding = self.actor_encoder(x)
            critic_embedding = self.critic_encoder(x)
        value = self.critic_head(critic_embedding)
        actor_output = self.actor_head(actor_embedding)
        if self.continuous:
            logit = [actor_output['mu'], actor_output['sigma']]
        else:
            logit = actor_output['logit']
        return {'logit': logit, 'value': value['pred']}

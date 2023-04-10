from typing import Union, Optional
from easydict import EasyDict
import torch
import torch.nn as nn
import treetensor.torch as ttorch
from copy import deepcopy
from ding.utils import SequenceType, squeeze
from ding.model.common import ReparameterizationHead, RegressionHead, MultiHead, \
    FCEncoder, ConvEncoder, IMPALAConvEncoder, PopArtVHead
from ding.torch_utils import MLP, fc_block


class DiscretePolicyHead(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        super(DiscretePolicyHead, self).__init__()
        self.main = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=nn.Linear,
                activation=activation,
                norm_type=norm_type
            ), fc_block(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PPOFModel(nn.Module):
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: str = 'discrete',
        share_encoder: bool = True,
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        fixed_sigma_value: Optional[int] = 0.3,
        bound_type: Optional[str] = None,
        encoder: Optional[torch.nn.Module] = None,
        popart_head=False,
    ) -> None:
        super(PPOFModel, self).__init__()
        obs_shape = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        self.share_encoder = share_encoder

        # Encoder Type
        def new_encoder(outsize):
            if isinstance(obs_shape, int) or len(obs_shape) == 1:
                return FCEncoder(
                    obs_shape=obs_shape,
                    hidden_size_list=encoder_hidden_size_list,
                    activation=activation,
                    norm_type=norm_type
                )
            elif len(obs_shape) == 3:
                return ConvEncoder(
                    obs_shape=obs_shape,
                    hidden_size_list=encoder_hidden_size_list,
                    activation=activation,
                    norm_type=norm_type
                )
            else:
                raise RuntimeError(
                    "not support obs_shape for pre-defined encoder: {}, please customize your own encoder".
                    format(obs_shape)
                )

        if self.share_encoder:
            assert actor_head_hidden_size == critic_head_hidden_size, \
                "actor and critic network head should have same size."
            if encoder:
                if isinstance(encoder, torch.nn.Module):
                    self.encoder = encoder
                else:
                    raise ValueError("illegal encoder instance.")
            else:
                self.encoder = new_encoder(actor_head_hidden_size)
        else:
            if encoder:
                if isinstance(encoder, torch.nn.Module):
                    self.actor_encoder = encoder
                    self.critic_encoder = deepcopy(encoder)
                else:
                    raise ValueError("illegal encoder instance.")
            else:
                self.actor_encoder = new_encoder(actor_head_hidden_size)
                self.critic_encoder = new_encoder(critic_head_hidden_size)

        # Head Type
        if not popart_head:
            self.critic_head = RegressionHead(
                critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
            )
        else:
            self.critic_head = PopArtVHead(
                critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
            )

        self.action_space = action_space
        assert self.action_space in ['discrete', 'continuous', 'hybrid'], self.action_space
        if self.action_space == 'continuous':
            self.multi_head = False
            self.actor_head = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type
            )
        elif self.action_space == 'discrete':
            actor_head_cls = DiscretePolicyHead
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
        elif self.action_space == 'hybrid':  # HPPO
            # hybrid action space: action_type(discrete) + action_args(continuous),
            # such as {'action_type_shape': torch.LongTensor([0]), 'action_args_shape': torch.FloatTensor([0.1, -0.27])}
            action_shape.action_args_shape = squeeze(action_shape.action_args_shape)
            action_shape.action_type_shape = squeeze(action_shape.action_type_shape)
            actor_action_args = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape.action_args_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                fixed_sigma_value=fixed_sigma_value,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type,
            )
            actor_action_type = DiscretePolicyHead(
                actor_head_hidden_size,
                action_shape.action_type_shape,
                actor_head_layer_num,
                activation=activation,
                norm_type=norm_type,
            )
            self.actor_head = nn.ModuleList([actor_action_type, actor_action_args])

        # must use list, not nn.ModuleList
        if self.share_encoder:
            self.actor = [self.encoder, self.actor_head]
            self.critic = [self.encoder, self.critic_head]
        else:
            self.actor = [self.actor_encoder, self.actor_head]
            self.critic = [self.critic_encoder, self.critic_head]
        # Convenient for calling some apis (e.g. self.critic.parameters()),
        # but may cause misunderstanding when `print(self)`
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: ttorch.Tensor, mode: str) -> ttorch.Tensor:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: ttorch.Tensor) -> ttorch.Tensor:
        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.actor_encoder(x)

        if self.action_space == 'discrete':
            return self.actor_head(x)
        elif self.action_space == 'continuous':
            x = self.actor_head(x)  # mu, sigma
            return ttorch.as_tensor(x)
        elif self.action_space == 'hybrid':
            action_type = self.actor_head[0](x)
            action_args = self.actor_head[1](x)
            return ttorch.as_tensor({'action_type': action_type, 'action_args': action_args})

    def compute_critic(self, x: ttorch.Tensor) -> ttorch.Tensor:
        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.critic_encoder(x)
        x = self.critic_head(x)
        return x

    def compute_actor_critic(self, x: ttorch.Tensor) -> ttorch.Tensor:
        if self.share_encoder:
            actor_embedding = critic_embedding = self.encoder(x)
        else:
            actor_embedding = self.actor_encoder(x)
            critic_embedding = self.critic_encoder(x)

        value = self.critic_head(critic_embedding)

        if self.action_space == 'discrete':
            logit = self.actor_head(actor_embedding)
            return ttorch.as_tensor({'logit': logit, 'value': value['pred']})
        elif self.action_space == 'continuous':
            x = self.actor_head(actor_embedding)
            return ttorch.as_tensor({'logit': x, 'value': value['pred']})
        elif self.action_space == 'hybrid':
            action_type = self.actor_head[0](actor_embedding)
            action_args = self.actor_head[1](actor_embedding)
            return ttorch.as_tensor(
                {
                    'logit': {
                        'action_type': action_type,
                        'action_args': action_args
                    },
                    'value': value['pred']
                }
            )

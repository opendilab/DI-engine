from typing import Union, Dict, Optional
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead
from ding.model.template.qac import QAC


@MODEL_REGISTRY.register('qac_diayn')
class QACDIAYN(QAC):
    r"""
    Overview:
        The QACDIAYN model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``, ``compute_discriminator``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_discriminator']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            num_skills: int,
            actor_head_type: str,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            discriminator_head_hidden_size: int = 100,
            discriminator_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the QACDIAYN Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - actor_head_type (:obj:`str`): Whether choose ``regression`` or ``reparameterization``.
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
        super(QACDIAYN, self).__init__(obs_shape, action_shape, actor_head_type)
        obs_shape: int = squeeze(obs_shape)
        num_skills: int = squeeze(num_skills)
        action_shape: int = squeeze(action_shape)
        self.actor_head_type = actor_head_type
        assert self.actor_head_type in ['regression', 'reparameterization']
        if self.actor_head_type == 'regression':
            self.actor = nn.Sequential(
                nn.Linear(obs_shape + num_skills, actor_head_hidden_size), activation,
                RegressionHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.actor_head_type == 'reparameterization':
            self.actor = nn.Sequential(
                nn.Linear(obs_shape + num_skills, actor_head_hidden_size), activation,
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
        if self.twin_critic:
            self.critic = nn.ModuleList()
            for _ in range(2):
                self.critic.append(
                    nn.Sequential(
                        nn.Linear(obs_shape + action_shape + num_skills, critic_head_hidden_size), activation,
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
                nn.Linear(obs_shape + action_shape + num_skills, critic_head_hidden_size),
                activation,
                RegressionHead(
                    critic_head_hidden_size,
                    1,  # output size
                    critic_head_layer_num,
                    final_tanh=False,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        self.discriminator = nn.Sequential(
            nn.Linear(obs_shape, discriminator_head_hidden_size),
            activation,
            RegressionHead(
                discriminator_head_hidden_size,
                num_skills,  # output size
                discriminator_head_layer_num,
                final_tanh=False,
                activation=activation,
                norm_type=norm_type
            )
        )


    def compute_discriminator(self, inputs: torch.Tensor) -> Dict:
        x = self.discriminator(inputs)['pred']
        return {'q_discriminator': x}

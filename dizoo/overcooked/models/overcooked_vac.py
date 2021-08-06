from typing import Union, Dict, Optional
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ding.torch_utils import fc_block, ResFCBlock, ResBlock
from ding.model.common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder


class BaselineConvEncoder(nn.Module):

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [25, 25, 25],
            activation: Optional[nn.Module] = nn.leakyRelu(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the Convolution Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, some ``output size``
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - activation (:obj:`nn.Module`):
                The type of activation to use in the conv ``layers`` and ``ResBlock``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResBlock`` for more details
        """
        super(BaselineConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list
        layers = []
        kernel_size = [5, 3, 3]
        stride = [1, 1, 1]
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], padding=(2, 2)))
            layers.append(self.act)
            input_size = hidden_size_list[i]
        assert len(set(hidden_size_list[3:-1])) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(3, len(self.hidden_size_list) - 1):
            layers.append(ResBlock(self.hidden_size_list[i], activation=self.act, norm_type=norm_type))
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.mid = nn.Linear(flatten_size, hidden_size_list[-1])

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Arguments:
            - x (:obj:`torch.Tensor`): Encoded Tensor after ``self.main``
        Returns:
            - outputs (:obj:`torch.Tensor`): Size int, also number of in-feature
        """
        test_data = torch.randn(1, *self.obs_shape)
        print("test_data = ", test_data)
        print("test data.shape = ", test_data.shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = x.float()
        x = self.main(x)
        x = self.mid(x)
        return x


@MODEL_REGISTRY.register('baselinevac_overcooked')
class BaselineVAC(nn.Module):
    r"""
    Overview:
        The Reproduced Baseline VAC model for overcook.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            share_encoder: bool = True,
            continuous: bool = False,
            encoder_hidden_size_list: SequenceType = [25, 25, 25],
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 3,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.leakyReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the VAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - share_encoder (:obj:`bool`): Whether share encoder.
            - continuous (:obj:`bool`): Whether collect continuously.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
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
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(BaselineVAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape: int = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        # Encoder Type

        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = BaselineConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        self.share_encoder = share_encoder
        self.share_encoder = True
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
        r"""
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N corresponding ``hidden_size``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        r"""
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
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
        r"""
        Shapes:
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.
        """
        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.critic_encoder(x)
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        r"""
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder.
            Returning the combination dictionry.
        """
        # print("origin input x shape is:", x.shape)
        if self.share_encoder:
            actor_embedding = critic_embedding = self.encoder(x)
        else:
            actor_embedding = self.actor_encoder(x)
            critic_embedding = self.critic_encoder(x)
        # print("critic_embedding is :", critic_embedding)
        # print("critic_embedding shape is :", critic_embedding.shape)
        value = self.critic_head(critic_embedding)
        actor_output = self.actor_head(actor_embedding)
        if self.continuous:
            logit = [actor_output['mu'], actor_output['sigma']]
        else:
            logit = actor_output['logit']
        return {'logit': logit, 'value': value['pred']}

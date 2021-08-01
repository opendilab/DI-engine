from typing import Union, Dict, Optional
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


@MODEL_REGISTRY.register('acer')
class ACER(nn.Module):
    r"""
    Overview:
        The ACER model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the ACER Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
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
        super(ACER, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape: int = squeeze(action_shape)
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )

        self.actor_encoder = encoder_cls(
            obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
        )
        self.critic_encoder = encoder_cls(
            obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
        )

        self.critic_head = RegressionHead(
            critic_head_hidden_size, action_shape, critic_head_layer_num, activation=activation, norm_type=norm_type
        )
        self.actor_head = DiscreteHead(
            actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type
        )
        self.actor = [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_head]
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
        Use observation to predict output.
        Parameter updates with ACER's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'``:
                - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.

            Forward with ``'compute_critic'``, inputs:`torch.Tensor` Necessary Keys:
                - ``obs`` encoded tensors.

            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.

                Forward with ``'compute_actor'``, Necessary Keys (either):
                    - logit (:obj:`torch.Tensor`):
                        - logit (:obj:`torch.Tensor`): Logit encoding tensor.

                Forward with ``'compute_critic'``, Necessary Keys:
                    - q_value (:obj:`torch.Tensor`): Q value tensor.

        Actor Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``

        Critic Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N1)`, B is batch size and N1 corresponds to ``obs_shape``
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
        Actor Examples:
            >>> # Regression mode
            >>> model = ACER(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 64])

        Critic Examples:
            >>> inputs = torch.randn(4,N)
            >>> model = ACER(obs_shape=(N, ),action_shape=5)
            >>> model(inputs, mode='compute_critic')['q_value'] # q value
            tensor([[-0.0681, -0.0431, -0.0530,  0.1454, -0.1093],
            [-0.0647, -0.0281, -0.0527,  0.1409, -0.1162],
            [-0.0596, -0.0321, -0.0676,  0.1386, -0.1113],
            [-0.0874, -0.0406, -0.0487,  0.1346, -0.1135]],
            grad_fn=<AddmmBackward>)


        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor) -> Dict:
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
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N1)`, where B is batch size and N1 is ``action_shape``
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N1)`, where B is batch size and N1 is ``action_shape``
        Examples:
            >>> # Regression mode
            >>> model = ACER(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 64])
        """
        x = self.actor_encoder(inputs)
        x = self.actor_head(x)

        return x

    def compute_critic(self, inputs: torch.Tensor) -> Dict:
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
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``.

        Examples:
            >>> inputs =torch.randn(4, N)
            >>> model = ACER(obs_shape=(N, ),action_shape=5)
            >>> model(inputs, mode='compute_critic')['q_value'] # q value
            tensor([[-0.0681, -0.0431, -0.0530,  0.1454, -0.1093],
            [-0.0647, -0.0281, -0.0527,  0.1409, -0.1162],
            [-0.0596, -0.0321, -0.0676,  0.1386, -0.1113],
            [-0.0874, -0.0406, -0.0487,  0.1346, -0.1135]],
            grad_fn=<AddmmBackward>)
        """

        obs = inputs
        x = self.critic_encoder(obs)
        x = self.critic_head(x)
        return {"q_value": x['pred']}

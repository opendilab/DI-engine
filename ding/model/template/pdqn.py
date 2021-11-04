from typing import Union, Optional, Dict
from easydict import EasyDict 

import torch
import torch.nn as nn

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, RegressionHead


@MODEL_REGISTRY.register('pdqn')
class PDQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: EasyDict,
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the PDQN (encoder + head) Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`EasyDict`): Action space shape in dict type, such as EasyDict({'action_type_shape': 3, 'action_args_shape': 5}).
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - dueling (:obj:`dueling`): Whether choose ``DuelingHead`` or ``DiscreteHead(default)``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details.
        """
        super(PDQN, self).__init__()
        
        # squeeze obs input for compatibility: 1, (1, ), [4, 32, 32]
        obs_shape = squeeze(obs_shape)
        # squeeze action shape input like (3,) to 3
        action_shape.action_args_shape = squeeze(action_shape.action_args_shape)
        action_shape.action_type_shape = squeeze(action_shape.action_type_shape)
        # init head hidden size
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]

        # Obs Encoder Type
        if isinstance(obs_shape, int) or len(obs_shape) == 1:  # FC Encoder
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        elif len(obs_shape) == 3:  # Conv Encoder
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )

        # Continuous Action Head Type
        self.cont_head = RegressionHead(
                    head_hidden_size,
                    action_shape.action_args_shape,
                    head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )

        # Discrete Action Head Type
        if dueling:
            dis_head_cls = DuelingHead
        else:
            dis_head_cls = DiscreteHead
        self.dis_head = dis_head_cls(
                head_hidden_size + action_shape.action_args_shape, 
                action_shape.action_type_shape, 
                head_layer_num, 
                activation=activation, 
                norm_type=norm_type
            )
        
        self.actor_head = nn.ModuleList([self.dis_head, self.cont_head])

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            PDQN forward computation graph, input observation tensor to predict q_value for discrete actions and values for continuous action_args
        Arguments:
            - x (:obj:`torch.Tensor`): Observation inputs
        Returns:
            - outputs (:obj:`Dict`): PDQN forward outputs, such as q_values and continuous action args.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each discrete action dimension.
            - action_args (:obj:`torch.FloatTensor`): Continuous action args (scaled from -1 to 1)
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is ``action_type_shape``
            - action_args (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where N2 is action_args_shape
        Examples:
            >>> action_shape = {'action_type_shape':3, 'action_args_type':5}
            >>> model = PDQN(32, action_shape)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 32)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict) 
            >>> assert outputs['logit'].shape == torch.Size([4, 3])
            >>> assert outputs['action_args'].shape == torch.Size([4,5])
        """
        x = self.encoder(x)  # size (B, encoded_state_shape)
        action_args = self.actor_head[1](x)  # size (B, action_args_shape)
        state_action_cat = torch.cat((x, action_args), dim=-1)  # size (B, encoded_state_shape + action_args_shape)
        logit = self.actor_head[0](state_action_cat)  # size (B, action_type_shape)
        return {'logit':logit['logit'], 'action_args': action_args['pred']}

from typing import Union, Optional, Dict
from easydict import EasyDict

import torch
import torch.nn as nn

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, RegressionHead


@MODEL_REGISTRY.register('pdqn')
class PDQN(nn.Module):
    mode = ['compute_discrete', 'compute_continuous']

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
        r"""
        Overview:
            Init the PDQN (encoder + head) Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`EasyDict`): Action space shape in dict type, such as \
                EasyDict({'action_type_shape': 3, 'action_args_shape': 5}).
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
            self.dis_encoder = FCEncoder(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
            self.cont_encoder = FCEncoder(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
        elif len(obs_shape) == 3:  # Conv Encoder
            self.dis_encoder = ConvEncoder(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
            self.cont_encoder = ConvEncoder(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own PDQN".format(obs_shape)
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
        # self.encoder = nn.ModuleList([self.dis_encoder, self.cont_encoder])
        self.encoder = nn.ModuleList([self.cont_encoder, self.cont_encoder])

    def forward(self, inputs: Union[torch.Tensor, Dict, EasyDict], mode: str) -> Dict:
        r"""
        Overview:
            PDQN forward computation graph, input observation tensor to predict q_value for \
            discrete actions and values for continuous action_args
        Arguments:
            - inputs (:obj:`torch.Tensor`): Observation inputs
            - mode (:obj:`str`): Name of the forward mode.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_continuous(self, inputs: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict continuous action args.
        Arguments:
            - inputs (:obj:`torch.Tensor`): Observation inputs
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
        Returns:
            - outputs (:obj:`Dict`): A dict with key 'action_args'
                -  'action_args': the continuous action args
        """
        cont_x = self.encoder[1](inputs)  # size (B, encoded_state_shape)
        action_args = self.actor_head[1](cont_x)['pred']  # size (B, action_args_shape)
        outputs = {'action_args': action_args}
        return outputs

    def compute_discrete(self, inputs: Union[Dict, EasyDict]) -> Dict:
        r"""
        Overview:
            Use observation tensor and continuous action args to predict discrete action types.
        Arguments:
            - inputs (:obj:`torch.Tensor`): A dict with keys 'state', 'action_args'
        Returns:
            - outputs (:obj:`Dict`): A dict with keys 'logit', 'action_args'
                -  'logit': the logit value for each discrete action,
                -  'action_args': the continuous action args(same as the inputs['action_args']) for later usage
        """
        dis_x = self.encoder[0](inputs['state'])  # size (B, encoded_state_shape)
        action_args = inputs['action_args']  # size (B, action_args_shape
        state_action_cat = torch.cat((dis_x, action_args), dim=-1)  # size (B, encoded_state_shape + action_args_shape)
        logit = self.actor_head[0](state_action_cat)['logit']  # size (B, action_type_shape)
        outputs = {'logit': logit, 'action_args': action_args}
        return outputs

from functools import partial
import os.path as osp

import torch
import torch.nn as nn

from ding.model import DuelingHead
from ding.config import read_config
from ding.utils import deep_merge_dicts, MODEL_REGISTRY
from ding.utils.data import default_collate
from ding.torch_utils import fc_block, Transformer, ResFCBlock, \
    conv2d_block, ResBlock, build_activation, ScatterConnection
import os
import yaml
from easydict import EasyDict
from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ding.model.common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead, RainbowHead, \
    QuantileHead, QRDQNHead, DistributionHead
from ding.model.template.q_learning import parallel_wrapper

with open(os.path.join(os.path.dirname(__file__), 'football_q_network_default_config.yaml')) as f:
    cfg = yaml.safe_load(f)
football_q_network_default_config = EasyDict(cfg)


@MODEL_REGISTRY.register('football_naive_q')
class FootballNaiveQ(nn.Module):

    def __init__(
            self,
            cfg: dict = {},
    ) -> None:
        super(FootballNaiveQ, self).__init__()
        self.cfg = deep_merge_dicts(football_q_network_default_config.model, cfg)
        scalar_encoder_arch = self.cfg.encoder.match_scalar
        player_encoder_arch = self.cfg.encoder.player
        self.scalar_encoder = ScalarEncoder(cfg=scalar_encoder_arch)
        self.player_type = player_encoder_arch.encoder_type
        assert self.player_type in ['transformer', 'spatial']
        if self.player_type == 'transformer':
            self.player_encoder = PlayerEncoder(cfg=player_encoder_arch.transformer)
        elif self.player_type == 'spatial':
            self.player_encoder = SpatialEncoder(cfg=player_encoder_arch.spatial)
        scalar_dim = self.scalar_encoder.output_dim
        player_dim = self.player_encoder.output_dim
        head_input_dim = scalar_dim + player_dim
        self.pred_head = FootballHead(input_dim=head_input_dim, cfg=self.cfg.policy)

    def forward(self, x: dict) -> dict:
        """
        Shape:
            - input: dict{obs_name: obs_tensor(:math: `(B, obs_dim)`)}
            - output: :math: `(B, action_dim)`
        """
        if isinstance(x, dict) and len(x) == 2:
            x = x['processed_obs']
        scalar_encodings = self.scalar_encoder(x)
        if self.player_type == 'transformer':
            player_encodings = self.player_encoder(x['players'], x['active_player'])
        elif self.player_type == 'spatial':
            player_encodings = self.player_encoder(x['players'])
        encoding_list = list(scalar_encodings.values()) + [player_encodings]
        x = torch.cat(encoding_list, dim=1)

        x = self.pred_head(x)
        # return x
        return {'logit': x, 'action': torch.argmax(x, dim=-1)}


@MODEL_REGISTRY.register('football_drqn')
class FootballDRQN(nn.Module):
    """
    Overview:
        DQN + RNN = DRQN
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType] = 1312,   # football
            action_shape: Union[int, SequenceType] = 19,  # football
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            lstm_type: Optional[str] = 'normal',
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            res_link: bool = False,
            env_name: Optional[str] = None,  # football
    ) -> None:
        r"""
        Overview:
            Init the DRQN Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - lstm_type (:obj:`Optional[str]`): Version of rnn cell, now support ['normal', 'pytorch', 'hpc', 'gru']
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
            - res_link (:obj:`bool`): use the residual link or not, default to False
        """
        super(FootballDRQN, self).__init__()

        # football related encoder
        self.cfg = deep_merge_dicts(football_q_network_default_config.model, cfg)
        scalar_encoder_arch = self.cfg.encoder.match_scalar
        player_encoder_arch = self.cfg.encoder.player
        self.scalar_encoder = ScalarEncoder(cfg=scalar_encoder_arch)
        self.player_type = player_encoder_arch.encoder_type
        assert self.player_type in ['transformer', 'spatial']
        if self.player_type == 'transformer':
            self.player_encoder = PlayerEncoder(cfg=player_encoder_arch.transformer)
        elif self.player_type == 'spatial':
            self.player_encoder = SpatialEncoder(cfg=player_encoder_arch.spatial)
        scalar_dim = self.scalar_encoder.output_dim
        player_dim = self.player_encoder.output_dim
        head_input_dim = scalar_dim + player_dim

        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        
        if env_name == 'football':
            # encoding shape: 1312 
            self.encoder = self.football_obs_encoder
            head_hidden_size = 1312
        else:
            # FC Encoder
            if isinstance(obs_shape, int) or len(obs_shape) == 1:
                self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
            # Conv Encoder
            elif len(obs_shape) == 3:
                self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
            else:
                raise RuntimeError(
                    "not support obs_shape for pre-defined encoder: {}, please customize your own DRQN".format(obs_shape)
                )
        # LSTM Type
        self.rnn = get_lstm(lstm_type, input_size=head_hidden_size, hidden_size=head_hidden_size)
        self.res_link = res_link
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    def football_obs_encoder(self, x):
        # football related encoder
        if isinstance(x, dict) and len(x) == 2:
            x = x['processed_obs']
        scalar_encodings = self.scalar_encoder(x)
        if self.player_type == 'transformer':
            player_encodings = self.player_encoder(x['players'], x['active_player'])
        elif self.player_type == 'spatial':
            player_encodings = self.player_encoder(x['players'])
        encoding_list = list(scalar_encodings.values()) + [player_encodings]
        x = torch.cat(encoding_list, dim=1)  
        # football obs encoding: shape (1312,)
        return x

    def forward(
            self, inputs: Dict, inference: bool = False, saved_hidden_state_timesteps: Optional[list] = None
    ) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict DRQN output.
            Parameter updates with DRQN's MLPs forward setup.
        Arguments:
            - inputs (:obj:`Dict`):
            - inference: (:obj:'bool'): if inference is True, we unroll the one timestep transition,
                if inference is False, we unroll the sequence transitions.
            - saved_hidden_state_timesteps: (:obj:'Optional[list]'): when inference is False,
                we unroll the sequence transitions, then we would save rnn hidden states at timesteps
                that are listed in list saved_hidden_state_timesteps.

       ArgumentsKeys:
            - obs (:obj:`torch.Tensor`): Encoded observation
            - prev_state (:obj:`list`): Previous state's tensor of size ``(B, N)``

        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``DRQN`` setups and return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``obs``.
            - next_state (:obj:`list`): Next state's tensor of size ``(B, N)``
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N=obs_space)`, where B is batch size.
            - prev_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - next_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`

        Examples:
            >>> # Init input's Keys:
            >>> prev_state = [[torch.randn(1, 1, 64) for __ in range(2)] for _ in range(4)] # B=4
            >>> obs = torch.randn(4,64)
            >>> model = DRQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=True)
            >>> # Check outputs's Keys
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == (4, 64)
            >>> assert len(outputs['next_state']) == 4
            >>> assert all([len(t) == 2 for t in outputs['next_state']])
            >>> assert all([t[0].shape == (1, 1, 64) for t in outputs['next_state']])
        """
        x, prev_state = inputs['obs'], inputs['prev_state']
        # for both inference and other cases, the network structure is encoder -> rnn network -> head
        # the difference is inference take the data with seq_len=1 (or T = 1)
        if inference:
            x = self.encoder(x)
            if self.res_link:
                a = x
            x = x.unsqueeze(0)  # for rnn input, put the seq_len of x as 1 instead of none.
            # prev_state: DataType: List[Tuple[torch.Tensor]]; Initially, it is a list of None
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)  # to delete the seq_len dim to match head network input
            if self.res_link:
                x = x + a
            x = self.head(x)
            x['next_state'] = next_state
            return x
        else:
            assert len(x.shape) in [3, 5], x.shape
            x = parallel_wrapper(self.encoder)(x)  # (T, B, N)
            if self.res_link:
                a = x
            lstm_embedding = []
            # TODO(nyz) how to deal with hidden_size key-value
            hidden_state_list = []
            if saved_hidden_state_timesteps is not None:
                saved_hidden_state = []
            for t in range(x.shape[0]):  # T timesteps
                output, prev_state = self.rnn(x[t:t + 1], prev_state)  # output: (1,B, head_hidden_size)
                if saved_hidden_state_timesteps is not None and t + 1 in saved_hidden_state_timesteps:
                    saved_hidden_state.append(prev_state)
                lstm_embedding.append(output)
                hidden_state = [p['h'] for p in prev_state]
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
                hidden_state_list.append(torch.cat(hidden_state, dim=1))
            x = torch.cat(lstm_embedding, 0)  # (T, B, head_hidden_size)
            if self.res_link:
                x = x + a
            x = parallel_wrapper(self.head)(x)  # (T, B, action_shape)
            # the last timestep state including h and c for lstm, {list: B{tuple: 2{Tensor:(1, 1, head_hidden_size}}}
            x['next_state'] = prev_state
            # all hidden state h, this returns a tensor of the dim: seq_len*batch_size*head_hidden_size
            # This key is used in qtran, the algorithm requires to retain all h_{t} during training
            x['hidden_state'] = torch.cat(hidden_state_list, dim=-3)
            if saved_hidden_state_timesteps is not None:
                x['saved_hidden_state'] = saved_hidden_state  # the selected saved hidden states, including h and c
            return x


class ScalarEncoder(nn.Module):

    def __init__(self, cfg: dict) -> None:
        super(ScalarEncoder, self).__init__()
        self.cfg = cfg
        self.act = nn.ReLU()
        self.output_dim = 0
        for k, arch in cfg.items():
            self.output_dim += arch['output_dim']
            encoder = fc_block(arch['input_dim'], arch['output_dim'], activation=self.act)
            setattr(self, k, encoder)

    def forward(self, x: dict) -> dict:
        """
        Shape:
            - input: dict{scalar_name: scalar_tensor(:math: `(B, scalar_dim)`)}
            - output: dict{scalar_name: scalar_encoded_tensor(:math: `(B, scalar_encoded_dim)`)}
        """
        fixed_scalar_sequence = [
            'ball_position', 'ball_direction', 'ball_rotation', 'ball_owned_team', 'ball_owned_player', 'active_player',
            'designated_player', 'active_player_sticky_actions', 'score', 'steps_left', 'game_mode'
        ]
        encodings = {}
        for k in fixed_scalar_sequence:
            data = x[k]
            # print(k, ' -- shape:{}, tensor:{}'.format(data.shape, data))
            encodings[k] = getattr(self, k)(data)
        return encodings


def cat_player_attr(player_data: dict) -> torch.Tensor:
    '''
    Arguments:
        player_data: {this_attr_name: [B, this_attr_dim]}
    Returns:
        attr: [B, total_attr_dim]
    '''
    fixed_player_attr_sequence = [
        'team', 'index', 'position', 'direction', 'tired_factor', 'yellow_card', 'active', 'role'
    ]
    attr = []
    for k in fixed_player_attr_sequence:
        if len(player_data[k].shape) == 1:
            player_data[k].unsqueeze_(-1)
        attr.append(player_data[k])
    attr = torch.cat(attr, dim=1)
    return attr


class PlayerEncoder(nn.Module):

    def __init__(
            self,
            cfg: dict,
    ) -> None:
        super(PlayerEncoder, self).__init__()
        self.act = nn.ReLU()
        self.player_num = cfg.player_num
        assert self.player_num in [1, 22], self.player_num
        self.output_dim = sum([dim for k, dim in cfg.player_attr_dim.items()]) * self.player_num
        player_transformer = Transformer(
            input_dim=cfg.input_dim,
            head_dim=cfg.head_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            head_num=cfg.head_num,
            mlp_num=cfg.mlp_num,
            layer_num=cfg.layer_num,
            dropout_ratio=cfg.dropout_ratio,
            activation=self.act,
        )
        setattr(self, 'players', player_transformer)

    def forward(self, x: list, active_player: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - input: list[len=22(=player_num/M)] -> element: dict{attr_name: attr_tensor(:math: `(B, attr_dim)`)}
            - active_player: :math: `(B, 11)`)
            - output: :math: `(B, player_num*total_attr_dim)`, player_num is in [1, 22]
        """
        player_input = self.get_player_input(x, active=active_player)  # (player_num*B, total_attr_dim)
        # player_output = getattr(self, 'players')(player_input, tensor_output=True)  # (player_num*B, total_attr_dim, 1)
        player_output = getattr(self, 'players')(player_input)  # (player_num*B, total_attr_dim, 1)
        player_output = player_output.squeeze(dim=2)  # (player_num*B, total_attr_dim)
        player_output = player_output.reshape((22, -1, player_output.shape[1]))  # (player_num, B, total_attr_dim)
        player_output = player_output.permute(1, 0, 2)  # (B, player_num, total_attr_dim)
        player_output = player_output.reshape((player_output.shape[0], -1))  # (B, player_num*total_attr_dim)
        return player_output

    def get_player_input(self, data: list, active: torch.Tensor) -> torch.Tensor:
        if self.player_num == 1:
            bs = data[0]['index'].shape[0]
            batch_player = [None for _ in range(bs)]
            for player in data:
                for idx in range(bs):
                    if batch_player[idx] is not None:
                        continue
                    if torch.nonzero(player['index'][idx]).item() == torch.nonzero(active[idx]).item() \
                            and torch.nonzero(player['team'][idx]).item() == 0:
                        batch_player[idx] = {k: v[idx] for k, v in player.items()}
                if None not in batch_player:
                    break
            # old_batch_player: list[len=bs] -> element: dict{attr_name: attr_tensor(:math: `(attr_dim)`)}
            batch_player = default_collate(batch_player)
            # new_batch_player: dict{attr_name: attr_tensor(:math: `(bs, attr_dim)`)}
            return cat_player_attr(batch_player).unsqueeze(dim=2)
        elif self.player_num == 22:
            players = []
            for player in data:
                players.append(cat_player_attr(player))
            players = torch.cat(players, dim=0)
            players = players.unsqueeze(dim=2)
            return players


class SpatialEncoder(nn.Module):

    def __init__(
            self,
            cfg: dict,
    ) -> None:
        super(SpatialEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.norm = cfg.norm_type
        self.scatter = ScatterConnection()
        input_dim = sum([dim for k, dim in cfg.player_attr_dim.items()])  # player_attr total dim
        self.project = conv2d_block(input_dim, cfg.project_dim, 1, 1, 0, activation=self.act, norm_type=self.norm)
        down_layers = []
        dims = [cfg.project_dim] + cfg.down_channels
        self.down_channels = cfg.down_channels
        for i in range(len(self.down_channels)):
            down_layers.append(nn.AvgPool2d(2, 2))
            down_layers.append(conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=self.norm))
        self.downsample = nn.Sequential(*down_layers)
        self.res = nn.ModuleList()
        dim = dims[-1]
        self.resblock_num = cfg.resblock_num
        for i in range(cfg.resblock_num):
            self.res.append(ResBlock(dim, dim, 3, 1, 1, activation=self.act, norm_type=self.norm))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_block(dim, cfg.fc_dim, activation=self.act)
        self.output_dim = cfg.fc_dim

    def forward(self, x: list) -> torch.Tensor:
        """
        Shape:
            - input: list[len=22(=player_num/M)] -> element: dict{attr_name: attr_tensor(:math: `(B, attr_dim)`)}
            - output: :math: `(B, fc_dim)`
        """
        players = []
        players_loc = []
        granularity = 0.01
        H, W = 84, 200
        for player in x:
            players.append(cat_player_attr(player))
            device = player['position'].device
            player_loc = ((player['position'] + torch.FloatTensor([1., 0.42]).to(device)) / granularity).long()
            player_loc_yx = player_loc[:, [1, 0]]
            players_loc.append(player_loc_yx)
        players = torch.stack(players, dim=1)  # [B, M, N]
        players_loc = torch.stack(players_loc, dim=1)  # [B, M, 2]
        players_loc[..., 0] = players_loc[..., 0].clamp(0, H - 1)
        players_loc[..., 1] = players_loc[..., 1].clamp(0, W - 1)
        x = self.scatter(players, (H, W), players_loc)
        x = self.project(x)
        x = self.downsample(x)
        for block in self.res:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.shape[:2])
        x = self.fc(x)
        return x


class FootballHead(nn.Module):

    def __init__(
            self,
            input_dim: int,
            cfg: dict,
    ) -> None:
        super(FootballHead, self).__init__()
        self.act = nn.ReLU()
        self.input_dim = input_dim
        self.hidden_dim = cfg.res_block.hidden_dim
        self.res_num = cfg.res_block.block_num
        self.dueling = cfg.dqn.dueling
        self.a_layer_num = cfg.dqn.a_layer_num
        self.v_layer_num = cfg.dqn.v_layer_num
        self.action_dim = cfg.action_dim
        self.pre_fc = fc_block(in_channels=input_dim, out_channels=self.hidden_dim, activation=self.act)
        res_blocks_list = []
        for i in range(self.res_num):
            res_blocks_list.append(ResFCBlock(in_channels=self.hidden_dim, activation=self.act, norm_type=None))
        self.res_blocks = nn.Sequential(*res_blocks_list)
        head_fn = partial(
            DuelingHead, a_layer_num=self.a_layer_num, v_layer_num=self.v_layer_num
        ) if self.dueling else nn.Linear
        self.pred = head_fn(self.hidden_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - input: :math: `(B, input_dim)`), input_dim is the sum of all encoders' output_dim
            - output: :math: `(B, action_dim)`)
        """
        x = self.pre_fc(x)
        x = self.res_blocks(x)
        x = self.pred(x)
        return x['logit']

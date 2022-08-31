import os
import sys
import random
import json
import copy
import enum
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dizoo.gfootball.model.bots.TamakEriFever.handyrl_core.model import BaseModel, Dense
from dizoo.gfootball.model.bots.TamakEriFever.football.util import *

# import dizoo.gfootball.model.TamakEriFever.football.rulebaseA as rulebaseA
# import dizoo.gfootball.model.TamakEriFever.football.rulebaseB as rulebaseB
# import dizoo.gfootball.model.TamakEriFever.football.rulebaseC as rulebaseC
# #import football.rulebaseD as rulebaseD
# import dizoo.gfootball.model.TamakEriFever.football.rulebaseE as rulebaseE
# import dizoo.gfootball.model.TamakEriFever.football.rulebaseF as rulebaseF


class MultiHeadAttention(nn.Module):
    # multi head attention for sets
    # https://github.com/akurniawan/pytorch-transformer/blob/master/modules/attention.py
    def __init__(self, in_dim, out_dim, out_heads, relation_dim=0, residual=False, projection=True, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_heads = out_heads
        self.relation_dim = relation_dim
        assert self.out_dim % self.out_heads == 0
        self.query_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim, bias=False)
        self.key_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim, bias=False)
        self.value_layer = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.residual = residual
        self.projection = projection
        if self.projection:
            self.proj_layer = nn.Linear(self.out_dim, self.out_dim)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln = nn.LayerNorm(self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.key_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)
        if self.projection:
            nn.init.uniform_(self.proj_layer.weight, -0.1, 0.1)

    def forward(self, query, key, relation=None, mask=None, key_mask=None, distance=None):
        """
        Args:
            query (torch.Tensor): [batch, query_len, in_dim]
            key (torch.Tensor): [batch, key_len, in_dim]
            relation (torch.Tensor): [batch, query_len, key_len, relation_dim]
            mask (torch.Tensor): [batch, query_len]
            key_mask (torch.Tensor): [batch, key_len]
        Returns:
            torch.Tensor: [batch, query_len, out_dim]
        """

        query_len = query.size(-2)
        key_len = key.size(-2)
        head_dim = self.out_dim // self.out_heads

        if key_mask is None:
            if torch.equal(query, key):
                key_mask = mask

        if relation is not None:
            relation = relation.view(-1, query_len, key_len, self.relation_dim)

            query_ = query.view(-1, query_len, 1, self.in_dim).repeat(1, 1, key_len, 1)
            query_ = torch.cat([query_, relation], dim=-1)

            key_ = key.view(-1, 1, key_len, self.in_dim).repeat(1, query_len, 1, 1)
            key_ = torch.cat([key_, relation], dim=-1)

            Q = self.query_layer(query_).view(-1, query_len * key_len, self.out_heads, head_dim)
            K = self.key_layer(key_).view(-1, query_len * key_len, self.out_heads, head_dim)

            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, key_len, head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, query_len, key_len, head_dim)

            attention = (Q * K).sum(dim=-1)
        else:
            Q = self.query_layer(query).view(-1, query_len, self.out_heads, head_dim)
            K = self.key_layer(key).view(-1, key_len, self.out_heads, head_dim)

            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, key_len, head_dim)

            attention = torch.bmm(Q, K.transpose(1, 2))

        if distance is not None:
            attention = attention - torch.log1p(distance.repeat(self.out_heads, 1, 1))
        attention = attention * (float(head_dim) ** -0.5)

        if key_mask is not None:
            attention = attention.view(-1, self.out_heads, query_len, key_len)
            attention = attention + ((1 - key_mask) * -1e32).view(-1, 1, 1, key_len)
        attention = F.softmax(attention, dim=-1)
        if mask is not None:
            attention = attention * mask.view(-1, 1, query_len, 1)
            attention = attention.contiguous().view(-1, query_len, key_len)

        V = self.value_layer(key).view(-1, key_len, self.out_heads, head_dim)
        V = V.transpose(1, 2).contiguous().view(-1, key_len, head_dim)

        output = torch.bmm(attention, V).view(-1, self.out_heads, query_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(*query.size()[:-2], query_len, self.out_dim)

        if self.projection:
            output = self.proj_layer(output)

        if self.residual:
            output = output + query

        if self.layer_norm:
            output = self.ln(output)

        if mask is not None:
            output = output * mask.unsqueeze(-1)
        attention = attention.view(*query.size()[:-2], self.out_heads, query_len, key_len).detach()

        return output, attention


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = nn.ReLU()  # activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class Conv2dAuto(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2
        )  # dynamic add padding based on the kernel_size


class ResNetResidualBlock(ResidualBlock):

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, partial(
            Conv2dAuto, kernel_size=3, bias=False
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        ) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def activation_func(activation):
    return nn.ModuleDict(
        [
            ['relu', nn.ReLU(inplace=True)], ['leaky_relu',
                                              nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)], ['none', nn.Identity()]
        ]
    )[activation]


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
    return nn.Sequential(conv3x3(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class FootballNet(BaseModel):

    class FootballEncoder(nn.Module):

        def __init__(self, filters):
            super().__init__()
            self.player_embedding = nn.Embedding(32, 5, padding_idx=0)
            self.mode_embedding = nn.Embedding(8, 3, padding_idx=0)
            self.fc_teammate = nn.Linear(23, filters)
            self.fc_opponent = nn.Linear(23, filters)
            self.fc = nn.Linear(filters + 41, filters)

        def forward(self, x):
            bs = x['mode_index'].size(0)
            # scalar features
            m_emb = self.mode_embedding(x['mode_index']).view(bs, -1)
            ball = x['ball']
            s = torch.cat([ball, x['match'], x['distance']['b2o'].view(bs, -1), m_emb], dim=1)

            # player features
            p_emb_self = self.player_embedding(x['player_index']['self'])
            ball_concat_self = ball.view(bs, 1, -1).repeat(1, x['player']['self'].size(1), 1)
            p_self = torch.cat([x['player']['self'], p_emb_self, ball_concat_self], dim=2)

            p_emb_opp = self.player_embedding(x['player_index']['opp'])
            ball_concat_opp = ball.view(bs, 1, -1).repeat(1, x['player']['opp'].size(1), 1)
            p_opp = torch.cat([x['player']['opp'], p_emb_opp, ball_concat_opp], dim=2)

            # encoding linear layer
            p_self = self.fc_teammate(p_self)
            p_opp = self.fc_opponent(p_opp)

            p = F.relu(torch.cat([p_self, p_opp], dim=1))
            s_concat = s.view(bs, 1, -1).repeat(1, p.size(1), 1)
            """
            TODO(pu): How to deal with dimension mismatch better?
            original code is:
            p = torch.cat([p, x['distance']['p2bo'].view(bs, p.size(1), -1), s_concat], dim=2)
            """
            p = torch.cat([p, x['distance']['p2bo'].repeat(1, 2, 1).view(bs, p.size(1), -1), s_concat], dim=2)
            h = F.relu(self.fc(p))

            # relation
            rel = None  # x['distance']['p2p']
            distance = None  # x['distance']['p2p']

            return h, rel, distance

    class FootballBlock(nn.Module):

        def __init__(self, filters, heads):
            super().__init__()
            self.attention = MultiHeadAttention(filters, filters, heads, relation_dim=0, residual=True, projection=True)

        def forward(self, x, rel, distance=None):
            h, _ = self.attention(x, x, relation=rel, distance=distance)
            return h

    class FootballControll(nn.Module):

        def __init__(self, filters, final_filters):
            super().__init__()
            self.filters = filters
            self.attention = MultiHeadAttention(filters, filters, 1, residual=False, projection=True)
            # self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)
            self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)

        def forward(self, x, e, control_flag):
            x_controled = (x * control_flag).sum(dim=1, keepdim=True)
            e_controled = (e * control_flag).sum(dim=1, keepdim=True)

            h, _ = self.attention(x_controled, x)

            h = torch.cat([x_controled, e_controled, h], dim=2).view(x.size(0), -1)
            # h = torch.cat([h, cnn_h.view(cnn_h.size(0), -1)], dim=1)
            h = self.fc_control(h)
            return h

    class FootballHead(nn.Module):

        def __init__(self, filters):
            super().__init__()
            self.head_p = nn.Linear(filters, 19, bias=False)
            self.head_p_special = nn.Linear(filters, 1 + 8 * 4, bias=False)
            self.head_v = nn.Linear(filters, 1, bias=True)
            self.head_r = nn.Linear(filters, 1, bias=False)

        def forward(self, x):
            p = self.head_p(x)
            p2 = self.head_p_special(x)
            v = self.head_v(x)
            r = self.head_r(x)
            return torch.cat([p, p2], -1), v, r

    class CNNModel(nn.Module):

        def __init__(self, final_filters):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(53, 128, kernel_size=1, stride=1, bias=False), nn.ReLU(inplace=True),
                nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False), nn.ReLU(inplace=True),
                nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False), nn.ReLU(inplace=True)
            )
            self.pool1 = nn.AdaptiveAvgPool2d((1, 11))
            self.conv2 = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 160, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(160),
                nn.Conv2d(160, 96, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.Conv2d(96, final_filters, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(final_filters),
            )
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = x['cnn_feature']
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            return x

    class SMMEncoder(nn.Module):

        class SMMBlock(nn.Module):

            def __init__(self, in_filters, out_filters, residuals=2):
                super().__init__()
                self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, bias=False)
                self.pool1 = nn.MaxPool2d(3, stride=2)
                self.blocks = nn.ModuleList([ResNetBasicBlock(out_filters, out_filters) for _ in range(residuals)])

            def forward(self, x):
                h = self.conv1(x)
                h = self.pool1(h)
                for block in self.blocks:
                    h = block(h)
                return h

        def __init__(self, filters):
            super().__init__()
            # 4, 72, 96 => filters, 1, 3
            self.blocks = nn.ModuleList(
                [
                    self.SMMBlock(4, filters),
                    self.SMMBlock(filters, filters),
                    self.SMMBlock(filters, filters),
                    self.SMMBlock(filters, filters),
                ]
            )

        def forward(self, x):
            x = x['smm']
            h = x
            for block in self.blocks:
                h = block(h)
            h = F.relu(h)
            return h

    class ActionHistoryEncoder(nn.Module):

        def __init__(self, input_size=19, hidden_size=64, num_layers=2, bidirectional=True):
            super().__init__()
            self.action_emd = nn.Embedding(19, 8)
            self.rnn = nn.GRU(8, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

        def forward(self, x):
            h = self.action_emd(x['action_history'])
            h = h.squeeze(dim=2)
            self.rnn.flatten_parameters()
            h, _ = self.rnn(h)
            return h

    def __init__(self, env, args={}, action_length=None):
        super().__init__(env, args, action_length)
        blocks = 5
        filters = 96
        final_filters = 128
        smm_filters = 32
        self.encoder = self.FootballEncoder(filters)
        self.blocks = nn.ModuleList([self.FootballBlock(filters, 8) for _ in range(blocks)])
        self.control = self.FootballControll(filters, final_filters)  # to head

        self.cnn = self.CNNModel(final_filters)  # to control
        # self.smm = self.SMMEncoder(smm_filters)  # to control
        rnn_hidden = 64
        self.rnn = self.ActionHistoryEncoder(19, rnn_hidden, 2)

        self.head = self.FootballHead(final_filters + final_filters + rnn_hidden * 2)
        # self.head = self.FootballHead(19, final_filters)

    def init_hidden(self, batch_size=None):
        return None

    def forward(self, x, hidden):
        e, rel, distance = self.encoder(x)
        h = e
        for block in self.blocks:
            h = block(h, rel, distance)
        cnn_h = self.cnn(x)
        # smm_h = self.smm(x)
        # h = self.control(h, e, x['control_flag'], cnn_h, smm_h)
        h = self.control(h, e, x['control_flag'])
        rnn_h = self.rnn(x)

        #         p, v, r = self.head(torch.cat([h,
        #                                        cnn_h.view(cnn_h.size(0), -1),
        #                                        smm_h.view(smm_h.size(0), -1)], axis=-1))

        rnn_h_head_tail = rnn_h[:, 0, :] + rnn_h[:, -1, :]
        rnn_h_plus_stick = torch.cat([rnn_h_head_tail[:, :-4], x['control']], dim=1)
        p, v, r = self.head(torch.cat([
            h,
            cnn_h.view(cnn_h.size(0), -1),
            rnn_h_plus_stick,
        ], axis=-1))
        # p, v, r = self.head(h)

        return p, torch.tanh(v), torch.tanh(r), hidden


OBS_TEMPLATE = {
    "controlled_players": 1,
    "players_raw": [
        {
            "right_team_active": [True, True, True, True, True, True, True, True, True, True, True],
            "right_team_yellow_card": [False, False, False, False, False, False, False, False, False, False, False],
            "left_team_tired_factor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "right_team_roles": [0, 2, 1, 1, 3, 5, 5, 5, 6, 9, 7],
            "left_team": [
                [-1.0110293626785278, -0.0], [-0.4266543984413147, -0.19894461333751678],
                [-0.5055146813392639, -0.06459399312734604], [-0.5055146813392639, 0.06459297984838486],
                [-0.4266543984413147, 0.19894461333751678], [-0.18624374270439148, -0.10739918798208237],
                [-0.270525187253952, -0.0], [-0.18624374270439148, 0.10739918798208237],
                [-0.010110294446349144, -0.21961550414562225], [-0.05055147036910057, -0.0],
                [-0.010110294446349144, 0.21961753070354462]
            ],
            "ball": [0.0, -0.0, 0.11061639338731766],
            "ball_owned_team": -1,
            "right_team_direction": [
                [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0],
                [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0]
            ],
            "left_team_direction": [
                [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0],
                [0.0, -0.0], [0.0, -0.0], [0.0, -0.0]
            ],
            "left_team_roles": [0, 2, 1, 1, 3, 5, 5, 5, 6, 9, 7],
            "score": [0, 0],
            "left_team_active": [True, True, True, True, True, True, True, True, True, True, True],
            "game_mode": 0,
            "steps_left": 3001,
            "ball_direction": [-0.0, 0.0, 0.006163952872157097],
            "ball_owned_player": -1,
            "right_team": [
                [1.0110293626785278, 0.0], [0.4266543984413147, 0.19894461333751678],
                [0.5055146813392639, 0.06459399312734604], [0.5055146813392639, -0.06459297984838486],
                [0.4266543984413147, -0.19894461333751678], [0.18624374270439148, 0.10739918798208237],
                [0.270525187253952, 0.0], [0.18624374270439148, -0.10739918798208237],
                [0.010110294446349144, 0.21961550414562225], [-0.0, -0.02032535709440708], [-0.0, 0.02032535709440708]
            ],
            "left_team_yellow_card": [False, False, False, False, False, False, False, False, False, False, False],
            "ball_rotation": [0.0, -0.0, 0.0],
            "right_team_tired_factor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "designated": 6,
            "active": 6,
            "sticky_actions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    ]
}

INFO_TEMPLATE = {'half_step': 1500}


# feature
def feature_from_states(states, info, player):
    # observation list to input tensor

    HISTORY_LENGTH = 8

    obs_history_ = [s[player]['observation']['players_raw'][0] for s in reversed(states[-HISTORY_LENGTH:])]
    obs_history = obs_history_ + [obs_history_[-1]] * (HISTORY_LENGTH - len(obs_history_))
    obs = obs_history[0]

    action_history_ = [s[player]['action'][0] for s in reversed(states[-HISTORY_LENGTH:])]
    action_history = action_history_ + [0] * (HISTORY_LENGTH - len(action_history_))
    """
    ・left players (x)
    ・left players (y)
    ・right players (x)
    ・right players (y)
    ・ball (x)
    ・ball (y)
    ・left goal (x)
    ・left goal (y)
    ・right goal (x)
    ・right goal (y)
    ・active (x)
    ・active (y)

    ・left players (x) - right players (x)
    ・left players (y) - right players (y)
    ・left players (x) - ball (x)
    ・left players (y) - ball (y)
    ・left players (x) - goal (x)
    ・left players (y) - goal (y)
    ・left players (x) - active (x)
    ・left players (y) - active (y)

    ・left players direction (x)
    ・left players direction (y)
    ・right players direction (x)
    ・right players direction (y)
    ・left players direction (x) - right players direction (x)
    ・left players direction (y) - right players direction (y)
    """

    # left players
    obs_left_team = np.array(obs['left_team'])
    left_player_x = np.repeat(obs_left_team[:, 0][..., None], 11, axis=1)
    left_player_y = np.repeat(obs_left_team[:, 1][..., None], 11, axis=1)

    # right players
    obs_right_team = np.array(obs['right_team'])
    right_player_x = np.repeat(obs_right_team[:, 0][..., None], 11, axis=1).transpose(1, 0)
    right_player_y = np.repeat(obs_right_team[:, 1][..., None], 11, axis=1).transpose(1, 0)

    # ball
    obs_ball = np.array(obs['ball'])
    ball_x = np.ones((11, 11)) * obs_ball[0]
    ball_y = np.ones((11, 11)) * obs_ball[1]
    ball_z = np.ones((11, 11)) * obs_ball[2]

    # goal
    left_goal, right_goal = [-1, 0], [1, 0]
    left_goal_x = np.ones((11, 11)) * left_goal[0]
    left_goal_y = np.ones((11, 11)) * left_goal[1]
    right_goal_x = np.ones((11, 11)) * right_goal[0]
    right_goal_y = np.ones((11, 11)) * right_goal[1]

    # side line
    side_line_y = [-.42, .42]
    side_line_y_top = np.ones((11, 11)) * side_line_y[0]
    side_line_y_bottom = np.ones((11, 11)) * side_line_y[1]

    # active
    active = np.array(obs['active'])
    active_player_x = np.repeat(obs_left_team[active][0][..., None, None], 11, axis=1).repeat(11, axis=0)
    active_player_y = np.repeat(obs_left_team[active][1][..., None, None], 11, axis=1).repeat(11, axis=0)

    # left players - right players
    left_minus_right_player_x = obs_left_team[:, 0][..., None] - obs_right_team[:, 0]
    left_minus_right_player_y = obs_left_team[:, 1][..., None] - obs_right_team[:, 1]

    # left players - ball
    left_minus_ball_x = (obs_left_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1)
    left_minus_ball_y = (obs_left_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1)

    # left players - right goal
    left_minus_right_goal_x = (obs_left_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1)
    left_minus_right_goal_y = (obs_left_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1)

    # left players - left goal
    left_minus_left_goal_x = (obs_left_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1)
    left_minus_left_goal_y = (obs_left_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1)

    # right players - right goal
    right_minus_right_goal_x = (obs_right_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_right_goal_y = (obs_right_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1).transpose(1, 0)

    # right players - left goal
    right_minus_left_goal_x = (obs_right_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_left_goal_y = (obs_right_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1).transpose(1, 0)

    # left players (x) - active
    left_minus_active_x = (obs_left_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11, axis=1)
    left_minus_active_y = (obs_left_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11, axis=1)

    # right player - ball
    right_minus_ball_x = (obs_right_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_ball_y = (obs_right_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1).transpose(1, 0)

    # right player - active
    right_minus_active_x = (obs_right_team[:, 0][..., None] - obs_left_team[active][0]).repeat(
        11, axis=1
    ).transpose(1, 0)
    right_minus_active_y = (obs_right_team[:, 1][..., None] - obs_left_team[active][1]).repeat(
        11, axis=1
    ).transpose(1, 0)

    # left player - side line
    left_minus_side_top = np.abs(obs_left_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1)
    left_minus_side_bottom = np.abs(obs_left_team[:, 1][..., None] - side_line_y[1]).repeat(11, axis=1)

    # right player - side line
    right_minus_side_top = np.abs(obs_right_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_side_bottom = np.abs(obs_right_team[:, 1][..., None] - side_line_y[1]).repeat(
        11, axis=1
    ).transpose(1, 0)

    # left players direction
    obs_left_team_direction = np.array(obs['left_team_direction'])
    left_player_direction_x = np.repeat(obs_left_team_direction[:, 0][..., None], 11, axis=1)
    left_player_direction_y = np.repeat(obs_left_team_direction[:, 1][..., None], 11, axis=1)

    # right players direction
    obs_right_team_direction = np.array(obs['right_team_direction'])
    right_player_direction_x = np.repeat(obs_right_team_direction[:, 0][..., None], 11, axis=1).transpose(1, 0)
    right_player_direction_y = np.repeat(obs_right_team_direction[:, 1][..., None], 11, axis=1).transpose(1, 0)

    # ball direction
    obs_ball_direction = np.array(obs['ball_direction'])
    ball_direction_x = np.ones((11, 11)) * obs_ball_direction[0]
    ball_direction_y = np.ones((11, 11)) * obs_ball_direction[1]
    ball_direction_z = np.ones((11, 11)) * obs_ball_direction[2]

    # left players direction - right players direction
    left_minus_right_player_direction_x = obs_left_team_direction[:, 0][..., None] - obs_right_team_direction[:, 0]
    left_minus_right_player_direction_y = obs_left_team_direction[:, 1][..., None] - obs_right_team_direction[:, 1]

    # left players direction - ball direction
    left_minus_ball_direction_x = (obs_left_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(11, axis=1)
    left_minus_ball_direction_y = (obs_left_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(11, axis=1)

    # right players direction - ball direction
    right_minus_ball_direction_x = (obs_right_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(
        11, axis=1
    ).transpose(1, 0)
    right_minus_ball_direction_y = (obs_right_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(
        11, axis=1
    ).transpose(1, 0)

    # ball rotation
    obs_ball_rotation = np.array(obs['ball_rotation'])
    ball_rotation_x = np.ones((11, 11)) * obs_ball_rotation[0]
    ball_rotation_y = np.ones((11, 11)) * obs_ball_rotation[1]
    ball_rotation_z = np.ones((11, 11)) * obs_ball_rotation[2]

    cnn_feature = np.stack(
        [
            left_player_x,
            left_player_y,
            right_player_x,
            right_player_y,
            ball_x,
            ball_y,
            ball_z,
            left_goal_x,
            left_goal_y,
            right_goal_x,
            right_goal_y,
            side_line_y_top,
            side_line_y_bottom,
            active_player_x,
            active_player_y,
            left_minus_right_player_x,
            left_minus_right_player_y,
            left_minus_right_goal_x,
            left_minus_right_goal_y,
            left_minus_left_goal_x,
            left_minus_left_goal_y,
            right_minus_right_goal_x,
            right_minus_right_goal_y,
            right_minus_left_goal_x,
            right_minus_left_goal_y,
            left_minus_side_top,
            left_minus_side_bottom,
            right_minus_side_top,
            right_minus_side_bottom,
            right_minus_ball_x,
            right_minus_ball_y,
            right_minus_active_x,
            right_minus_active_y,
            left_minus_ball_x,
            left_minus_ball_y,
            left_minus_active_x,
            left_minus_active_y,
            ball_direction_x,
            ball_direction_y,
            ball_direction_z,
            left_minus_ball_direction_x,
            left_minus_ball_direction_y,
            right_minus_ball_direction_x,
            right_minus_ball_direction_y,
            left_player_direction_x,
            left_player_direction_y,
            right_player_direction_x,
            right_player_direction_y,
            left_minus_right_player_direction_x,
            left_minus_right_player_direction_y,
            ball_rotation_x,
            ball_rotation_y,
            ball_rotation_z,
        ],
        axis=0
    )

    # ball
    BALL_OWEND_1HOT = {-1: [0, 0], 0: [1, 0], 1: [0, 1]}
    ball_owned_team_ = obs['ball_owned_team']
    ball_owned_team = BALL_OWEND_1HOT[ball_owned_team_]  # {-1, 0, 1} None, self, opponent
    PLAYER_1HOT = np.concatenate([np.eye(11), np.zeros((1, 11))])
    ball_owned_player_ = PLAYER_1HOT[obs['ball_owned_player']]  # {-1, N-1}
    if ball_owned_team_ == -1:
        my_ball_owned_player = PLAYER_1HOT[-1]
        op_ball_owned_player = PLAYER_1HOT[-1]
    elif ball_owned_team_ == 0:
        my_ball_owned_player = ball_owned_player_
        op_ball_owned_player = PLAYER_1HOT[-1]
    else:
        my_ball_owned_player = PLAYER_1HOT[-1]
        op_ball_owned_player = ball_owned_player_

    ball_features = np.concatenate([obs['ball'], obs['ball_direction'], obs['ball_rotation']]).astype(np.float32)

    # self team
    left_team_features = np.concatenate(
        [
            [[1] for _ in obs['left_team']],  # left team flag
            obs['left_team'],  # position
            obs['left_team_direction'],
            [[v] for v in obs['left_team_tired_factor']],
            [[v] for v in obs['left_team_yellow_card']],
            [[v] for v in obs['left_team_active']],
            my_ball_owned_player[..., np.newaxis]
        ],
        axis=1
    ).astype(np.float32)

    left_team_indice = np.arange(0, 11, dtype=np.int32)

    # opponent team
    right_team_features = np.concatenate(
        [
            [[0] for _ in obs['right_team']],  # right team flag
            obs['right_team'],  # position
            obs['right_team_direction'],
            [[v] for v in obs['right_team_tired_factor']],
            [[v] for v in obs['right_team_yellow_card']],
            [[v] for v in obs['right_team_active']],
            op_ball_owned_player[..., np.newaxis]
        ],
        axis=1
    ).astype(np.float32)

    right_team_indice = np.arange(0, 11, dtype=np.int32)

    # distance information
    def get_distance(xy1, xy2):
        return (((xy1 - xy2) ** 2).sum(axis=-1)) ** 0.5

    def get_line_distance(x1, x2):
        return np.abs(x1 - x2)

    def multi_scale(x, scale):
        return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

    both_team = np.array(obs['left_team'] + obs['right_team'], dtype=np.float32)
    ball = np.array([obs['ball'][:2]], dtype=np.float32)
    goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
    goal_line_x = np.array([-1, 1], dtype=np.float32)
    side_line_y = np.array([-.42, .42], dtype=np.float32)

    # ball <-> goal, goal line, side line distance
    b2g_distance = get_distance(ball, goal)
    b2gl_distance = get_line_distance(ball[0][0], goal_line_x)
    b2sl_distance = get_line_distance(ball[0][1], side_line_y)
    b2o_distance = np.concatenate([b2g_distance, b2gl_distance, b2sl_distance], axis=-1)

    # player <-> ball, goal, back line, side line distance
    p2b_distance = get_distance(both_team[:, np.newaxis, :], ball[np.newaxis, :, :])
    p2g_distance = get_distance(both_team[:, np.newaxis, :], goal[np.newaxis, :, :])
    p2gl_distance = get_line_distance(both_team[:, :1], goal_line_x[np.newaxis, :])
    p2sl_distance = get_line_distance(both_team[:, 1:], side_line_y[np.newaxis, :])
    p2bo_distance = np.concatenate([p2b_distance, p2g_distance, p2gl_distance, p2sl_distance], axis=-1)

    # player <-> player distance
    p2p_distance = get_distance(both_team[:, np.newaxis, :], both_team[np.newaxis, :, :])

    # apply Multiscale to distances
    # def concat_multiscale(x, scale):
    #    return np.concatenate([x[...,np.newaxis], 1 - multi_scale(x, scale)], axis=-1)

    # distance_scales = [.01, .05, .25, 1.25]
    # b2o_distance = 1 - multi_scale(b2o_distance, distance_scales).reshape(-1)
    # p2bo_distance = 1 - multi_scale(p2bo_distance, distance_scales).reshape(len(both_team), -1)
    # p2p_distance = 1 - multi_scale(p2p_distance, distance_scales).reshape(len(both_team), len(both_team), -1)

    # controlled player information
    control_flag_ = np.array(PLAYER_1HOT[obs['active']], dtype=np.float32)
    control_flag = np.concatenate([control_flag_, np.zeros(len(obs['right_team']))])[..., np.newaxis]

    # controlled status information
    DIR = [
        [-1, 0],
        [-.707, -.707],
        [0, 1],
        [.707, -.707],  # L, TL, T, TR
        [1, 0],
        [.707, .707],
        [0, -1],
        [-.707, .707]  # R, BR, B, BL
    ]
    sticky_direction = DIR[obs['sticky_actions'][:8].index(1)] if 1 in obs['sticky_actions'][:8] else [0, 0]
    sticky_flags = obs['sticky_actions'][8:]

    control_features = np.concatenate([
        sticky_direction,
        sticky_flags,
    ]).astype(np.float32)

    # Match state
    if obs['steps_left'] > info['half_step']:
        steps_left_half = obs['steps_left'] - info['half_step']
    else:
        steps_left_half = obs['steps_left']
    match_features = np.concatenate(
        [
            multi_scale(obs['score'], [1, 3]).ravel(),
            multi_scale(obs['score'][0] - obs['score'][1], [1, 3]),
            multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),
            multi_scale(steps_left_half, [10, 100, 1000, 10000]),
            ball_owned_team,
        ]
    ).astype(np.float32)

    mode_index = np.array([obs['game_mode']], dtype=np.int32)

    # Super Mini Map
    # SMM_WIDTH = 96 #// 3
    # SMM_HEIGHT = 72 #// 3
    # SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

    # # Normalized minimap coordinates
    # MINIMAP_NORM_X_MIN = -1.0
    # MINIMAP_NORM_X_MAX = 1.0
    # MINIMAP_NORM_Y_MIN = -1.0 / 2.25
    # MINIMAP_NORM_Y_MAX = 1.0 / 2.25

    # _MARKER_VALUE = 1  # 255

    # def get_smm_layers(config):
    #     return SMM_LAYERS

    # def mark_points(frame, points):
    #     """Draw dots corresponding to 'points'.
    #     Args:
    #       frame: 2-d matrix representing one SMM channel ([y, x])
    #       points: a list of (x, y) coordinates to be marked
    #     """
    #     for p in range(len(points) // 2):
    #         x = int((points[p * 2] - MINIMAP_NORM_X_MIN) /
    #                 (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1])
    #         y = int((points[p * 2 + 1] - MINIMAP_NORM_Y_MIN) /
    #                 (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0])
    #         x = max(0, min(frame.shape[1] - 1, x))
    #         y = max(0, min(frame.shape[0] - 1, y))
    #         frame[y, x] = _MARKER_VALUE

    # def generate_smm(observation, config=None,
    #                  channel_dimensions=(SMM_WIDTH, SMM_HEIGHT)):
    #     """Returns a list of minimap observations given the raw features for each
    #     active player.
    #     Args:
    #       observation: raw features from the environment
    #       config: environment config
    #       channel_dimensions: resolution of SMM to generate
    #     Returns:
    #       (N, H, W, C) - shaped np array representing SMM. N stands for the number of
    #       players we are controlling.
    #     """
    #     frame = np.zeros((len(observation), channel_dimensions[1],
    #                       channel_dimensions[0], len(get_smm_layers(config))),
    #                       dtype=np.uint8)

    #     for o_i, o in enumerate(observation):
    #         for index, layer in enumerate(get_smm_layers(config)):
    #             assert layer in o
    #             if layer == 'active':
    #                 if o[layer] == -1:
    #                     continue
    #                 mark_points(frame[o_i, :, :, index],
    #                             np.array(o['left_team'][o[layer]]).reshape(-1))
    #             else:
    #                 mark_points(frame[o_i, :, :, index], np.array(o[layer]).reshape(-1))
    #     return frame

    # smm = generate_smm([obs]).transpose(3, 1, 2, 0).squeeze(3).astype(np.float32)

    # ACTION_1HOT = np.eye(19)
    # action_history = np.stack([ACTION_1HOT[a] for a in action_history]).astype(np.float32)
    action_history = np.array(action_history, dtype=np.int32)[..., None]

    return {
        # features
        'ball': ball_features,
        'match': match_features,
        'player': {
            'self': left_team_features,
            'opp': right_team_features
        },
        'control': control_features,
        'player_index': {
            'self': left_team_indice,
            'opp': right_team_indice
        },
        'mode_index': mode_index,
        'control_flag': control_flag,
        # distances
        'distance': {
            'p2p': p2p_distance,
            'p2bo': p2bo_distance,
            'b2o': b2o_distance
        },
        # CNN
        'cnn_feature': cnn_feature,
        # SuperMiniMap
        # 'smm': smm,
        'action_history': action_history
    }


KICK_ACTIONS = {
    Action.LongPass: 20,
    Action.HighPass: 28,
    Action.ShortPass: 36,
    Action.Shot: 44,
}


class Environment:
    ACTION_LEN = 19 + 4 * 8
    ACTION_IDX = list(range(ACTION_LEN))

    def __init__(self, args={}):
        self.env_map = {}
        self.env = None
        self.limit_steps = args.get('limit_steps', 100000)
        self.frame_skip = args.get('frame_skip', 0)
        self.reset_common()

    def reset_common(self):
        self.finished = False
        self.prev_score = [0, 0]
        self.reset_flag = False
        self.checkpoint = [
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
        ]
        self.states = []
        self.half_step = 1500
        self.reserved_action = [None, None]

    def reset(self, args={}):
        if len(self.env_map) == 0:
            from gfootball.env import football_action_set
            from gfootball.env.wrappers import Simple115StateWrapper
            from kaggle_environments import make

            self.ACTION_STR = football_action_set.action_set_v1
            self.ACTION2STR = {i: j for i, j in enumerate(football_action_set.action_set_v1)}
            self.STR2ACTION = {j: i for i, j in self.ACTION2STR.items()}

            #             self.env_map[3000] = make("football", configuration={"scenario_name": "11_vs_11_kaggle"})
            #             self.env_map[1000] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_1000_500"})
            #             self.env_map[500] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_500_250"})
            #             self.env_map[9999] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_random"})
            #             self.env_map[99999] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_random_long"})

            self.env_map["real"] = make("football", configuration={"scenario_name": "11_vs_11_kaggle"})
            self.env_map["eval"] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_1000_500"})
            self.env_map["train"] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_train"})

        # decide limit steps

        #         if args.get('role', {}) == 'e':
        #             self.env = self.env_map[1000]
        #         else:
        #             limit_rate = args.get('limit_rate', 1.0)
        #             if limit_rate > 0.9:
        #                 self.env = self.env_map[3000]
        #             elif limit_rate >= 0:
        #                 self.env = self.env_map[99999]

        role = args.get('role', '')
        limit_rate = args.get('limit_rate', 1)
        if role == 'g':
            self.env = self.env_map['train' if limit_rate < 0.95 else 'real']
        elif role == 'e':
            self.env = self.env_map['eval']
        else:
            self.env = self.env_map['real']

        state = self.env.reset()
        self.resets_info(state)

    def resets_info(self, state):
        self.reset_common()
        state = copy.deepcopy(state)
        state = [self._preprocess_state(s) for s in state]
        self.states.append(state)
        self.half_step = state[0]['observation']['players_raw'][0]['steps_left'] // 2

    def reset_info(self, state):
        self.resets_info(state)

    def chance(self):
        pass

    def action2str(self, a: int):
        # return self.ACTION2STR[a]
        return str(a)

    def str2action(self, s: str):
        # return self.STR2ACTION[s]
        return int(s)

    def plays(self, actions):
        self._plays(actions)

    def _plays(self, actions):
        # state transition function
        # action is integer (0 ~ 18)
        actions = copy.deepcopy(actions)
        for i, res_action in enumerate(self.reserved_action):
            if res_action is not None:
                actions[i] = res_action

        # augmented action to atomic action
        for i, action in enumerate(actions):
            atomic_a, reserved_a = self.special_to_actions(action)
            actions[i] = atomic_a
            self.reserved_action[i] = reserved_a

        # step environment
        state = self.env.step([[actions[0]], [actions[1]]])
        state = copy.deepcopy(state)
        state = [self._preprocess_state(s) for s in state]
        self.states.append(state)

        # update status
        if state[0]['status'] == 'DONE' or len(self.states) > self.limit_steps:
            self.finished = True

    def plays_info(self, state):
        # state stansition function as an agent
        state = copy.deepcopy(state)
        state = [self._preprocess_state(s) for s in state]
        self.states.append(state)

    def play_info(self, state):
        self.plays_info(state)

    def diff_info(self):
        return self.states[-1]

    def turns(self):
        return self.players()

    def players(self):
        return [0, 1]

    def terminal(self):
        # check whether the state is terminal
        return self.finished

    def reward(self):
        prev_score = self.prev_score
        score = self.score()

        rs = []
        scored_player = None
        for p in self.players():
            r = 1.0 * (score[p] - prev_score[p]) - 1.0 * (score[1 - p] - prev_score[1 - p])
            rs.append(r)
            if r != 0:
                self.reset_flag = True
                scored_player = p

        self.prev_score = self.score()
        return rs

        def get_goal_distance(xy1):
            return (((xy1 - np.array([1, 0])) ** 2).sum(axis=-1)) ** 0.5

        # checkpoint reward (https://arxiv.org/pdf/1907.11180.pdf)
        checkpoint_reward = []
        for p in self.players():
            obs = self.raw_observation(p)['players_raw'][0]
            ball_owned_team = obs['ball_owned_team']
            if ball_owned_team == p and len(self.checkpoint[p]) != 0:
                ball = obs['ball'][:2]
                goal_distance = get_goal_distance(ball)
                if goal_distance < self.checkpoint[p][0]:
                    cr = 0
                    for idx, c in enumerate(self.checkpoint[p]):
                        if goal_distance < c:
                            cr += 0.1
                        else:
                            break
                    self.checkpoint[p] = self.checkpoint[p][idx:]
                    checkpoint_reward.append(cr)
                else:
                    checkpoint_reward.append(0)
            else:
                checkpoint_reward.append(0)

        if scored_player is not None:
            checkpoint_reward[scored_player] += len(
                self.checkpoint[scored_player]
            ) * 0.1  # add remain reward when scoring (0.05 per checkpoint)
            self.checkpoint[scored_player] = []

        return [rs[p] + checkpoint_reward[p] for p in self.players()]

    def is_reset_state(self):
        if self.reset_flag:
            self.reset_flag = False
            return True
        return False

    def score(self):
        if len(self.states) == 0:
            return [0, 0]
        obs = self.states[-1]
        return [
            obs[0]['observation']['players_raw'][0]['score'][0], obs[1]['observation']['players_raw'][0]['score'][0]
        ]

    def outcome(self):
        if len(self.states) == 0:
            return [0, 0]
        scores = self.score()
        if scores[0] > scores[1]:
            score_diff = scores[0] - scores[1]
            outcome_tanh = np.tanh(score_diff ** 0.8)
            return [outcome_tanh, -outcome_tanh]
        elif scores[0] < scores[1]:
            score_diff = scores[1] - scores[0]
            outcome_tanh = np.tanh(score_diff ** 0.8)
            return [-outcome_tanh, outcome_tanh]
        return [0, 0]

    def legal_actions(self, player):
        # legal action list
        all_actions = [i for i in copy.copy(self.ACTION_IDX) if i != 19]

        if len(self.states) == 0:
            return all_actions

        # obs from view of the player
        obs = self.raw_observation(player)['players_raw'][0]
        # Illegal actions
        illegal_actions = set()
        # You have a ball?
        ball_owned_team = obs['ball_owned_team']
        if ball_owned_team != 0:  # not owned or free
            illegal_actions.add(int(Action.LongPass))
            illegal_actions.add(int(Action.HighPass))
            illegal_actions.add(int(Action.ShortPass))
            illegal_actions.add(int(Action.Shot))
            illegal_actions.add(int(Action.Dribble))
            for d in range(8):
                illegal_actions.add(KICK_ACTIONS[Action.LongPass] + d)
                illegal_actions.add(KICK_ACTIONS[Action.HighPass] + d)
                illegal_actions.add(KICK_ACTIONS[Action.ShortPass] + d)
                illegal_actions.add(KICK_ACTIONS[Action.Shot] + d)
        else:  # owned
            illegal_actions.add(int(Action.Slide))

        # Already sticky action?
        sticky_actions = obs['sticky_actions']
        if type(sticky_actions) == set:
            sticky_actions = [0] * 10

        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
            illegal_actions.add(int(Action.ReleaseSprint))

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
            illegal_actions.add(int(Action.ReleaseDribble))

        if 1 not in sticky_actions[:8]:
            illegal_actions.add(int(Action.ReleaseDirection))

        return [a for a in all_actions if a not in illegal_actions]

    def action_length(self):
        # maximum size of policy (it determines output size of policy function)
        return self.ACTION_LEN

    def raw_observation(self, player):
        if len(self.states) > 0:
            return self.states[-1][player]['observation']
        else:
            return OBS_TEMPLATE

    def observation(self, player):
        # input feature for neural nets
        info = {'half_step': self.half_step}
        return feature_from_states(self.states, info, player)

    def _preprocess_state(self, player_state):
        if player_state is None:
            return player_state

        # in ball-dead state, set ball owned player and team
        o = player_state['observation']['players_raw'][0]
        mode = o['game_mode']
        if mode == GameMode.FreeKick or \
                mode == GameMode.Corner or \
                mode == GameMode.Penalty or \
                mode == GameMode.GoalKick:
            # find nearest player and team
            def dist(xy1, xy2):
                return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

            team_player_position = [(0, i, p) for i, p in enumerate(o['left_team'])] + \
                                   [(1, i, p) for i, p in enumerate(o['right_team'])]
            distances = [(t[0], t[1], dist(t[2], o['ball'][:2])) for t in team_player_position]
            distances = sorted(distances, key=lambda x: x[2])
            # print(mode, [t[2] for t in distances])
            # print(o['ball_owned_team'], o['ball_owned_player'], '->', distances[0][0], distances[0][1])
            # input()
            o['ball_owned_team'] = distances[0][0]
            o['ball_owned_player'] = distances[0][1]

        # in the beginning, fill actions with 0
        if len(player_state['action']) == 0:
            player_state['action'].append(0)

        return player_state

    def special_to_actions(self, saction):
        if not 0 <= saction < 52:
            return [0, None]
        for a, index in KICK_ACTIONS.items():
            if index <= saction < index + 8:
                return [a, Action(saction - index + 1)]
        return [saction, None]

    '''def action_to_specials(self, action):
        p = np.zeros(self.action_length())
        p[action] = 1

        sticky_direction =


        if action == Action.LongPass:
            return

        return p / p.sum()'''

    def funcname(self, parameter_list):
        """
        docstring
        """
        pass

    def net(self):
        return FootballNet

    def rule_based_action(self, player):
        return 19

    # def rule_based_action_A(self, player):
    #     return rulebaseA._agent(self.states[-1][player]['observation'])

    # def rule_based_action_B(self, player):
    #     return rulebaseB._agent(self.states[-1][player]['observation'])

    # def rule_based_action_C(self, player):
    #     return rulebaseC._agent(self.states[-1][player]['observation'])

    # #def rule_based_action_D(self, player):
    # #    return rulebaseD._agent(self.states[-1][player]['observation'])

    # def rule_based_action_E(self, player):
    #     return rulebaseE._agent(self.states[-1][player]['observation'])

    # def rule_based_action_F(self, player):
    #     return rulebaseF._agent(self.states[-1][player]['observation'])


if __name__ == '__main__':
    e = Environment()
    net = e.net()(e)
    net.eval()
    for _ in range(1):
        e.reset()
        o = e.observation(0)
        net.inference(o, None)
        while not e.terminal():
            # print(e)
            _ = e.observation(0)
            _ = e.observation(1)
            print(e.env.configuration.episodeSteps)
            print(e.raw_observation(0)['players_raw'][0]['steps_left'])
            action_list = [0, 0]
            action_list[0] = random.choice(e.legal_actions(0))
            action_list[1] = e.rule_based_action_C(1)
            print(len(e.states), action_list)
            e.plays(action_list)
            print(e.checkpoint)
            print(e.reward())
        print(e)
        print(e.score())
        print(e.outcome())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ding.utils import MODEL_REGISTRY, deep_merge_dicts
from ding.config import read_config
from dizoo.gfootball.model.conv1d.conv1d_default_config import conv1d_default_config


@MODEL_REGISTRY.register('conv1d')
class GfootballConv1DModel(nn.Module):

    def __init__(
            self,
            cfg: dict = {},
    ) -> None:
        super(GfootballConv1DModel, self).__init__()
        self.cfg = deep_merge_dicts(conv1d_default_config, cfg)

        self.fc_player = nn.Linear(
            self.cfg.feature_embedding.player.input_dim, self.cfg.feature_embedding.player.output_dim
        )
        self.fc_ball = nn.Linear(self.cfg.feature_embedding.ball.input_dim, self.cfg.feature_embedding.ball.output_dim)
        self.fc_left = nn.Linear(
            self.cfg.feature_embedding.left_team.input_dim, self.cfg.feature_embedding.left_team.output_dim
        )
        self.fc_right = nn.Linear(
            self.cfg.feature_embedding.right_team.input_dim, self.cfg.feature_embedding.right_team.output_dim
        )
        self.fc_left_closest = nn.Linear(
            self.cfg.feature_embedding.left_closest.input_dim, self.cfg.feature_embedding.left_closest.output_dim
        )
        self.fc_right_closest = nn.Linear(
            self.cfg.feature_embedding.right_closest.input_dim, self.cfg.feature_embedding.right_closest.output_dim
        )

        self.conv1d_left = nn.Conv1d(
            self.cfg.feature_embedding.left_team.output_dim,
            self.cfg.feature_embedding.left_team.conv1d_output_channel,
            1,
            stride=1
        )
        self.conv1d_right = nn.Conv1d(
            self.cfg.feature_embedding.right_team.output_dim,
            self.cfg.feature_embedding.right_team.conv1d_output_channel,
            1,
            stride=1
        )
        self.fc_left2 = nn.Linear(
            self.cfg.feature_embedding.left_team.conv1d_output_channel * 10,
            self.cfg.feature_embedding.left_team.fc_output_dim
        )
        self.fc_right2 = nn.Linear(
            self.cfg.feature_embedding.right_team.conv1d_output_channel * 11,
            self.cfg.feature_embedding.right_team.fc_output_dim
        )
        self.fc_cat = nn.Linear(self.cfg.fc_cat.input_dim, self.cfg.lstm_size)

        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(48)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_left_closest = nn.LayerNorm(48)
        self.norm_right = nn.LayerNorm(48)
        self.norm_right2 = nn.LayerNorm(96)
        self.norm_right_closest = nn.LayerNorm(48)
        self.norm_cat = nn.LayerNorm(self.cfg.lstm_size)

        self.lstm = nn.LSTM(self.cfg.lstm_size, self.cfg.lstm_size)

        self.fc_pi_a1 = nn.Linear(self.cfg.lstm_size, self.cfg.policy_head.hidden_dim)
        self.fc_pi_a2 = nn.Linear(self.cfg.policy_head.hidden_dim, self.cfg.policy_head.act_shape)
        self.norm_pi_a1 = nn.LayerNorm(164)

        self.fc_pi_m1 = nn.Linear(self.cfg.lstm_size, 164)
        self.fc_pi_m2 = nn.Linear(164, 8)
        self.norm_pi_m1 = nn.LayerNorm(164)

        self.fc_v1 = nn.Linear(self.cfg.lstm_size, self.cfg.value_head.hidden_dim)
        self.norm_v1 = nn.LayerNorm(164)
        self.fc_v2 = nn.Linear(self.cfg.value_head.hidden_dim, self.cfg.value_head.output_dim, bias=False)

    def forward(self, state_dict):
        player_state = state_dict["player"].unsqueeze(0)
        ball_state = state_dict["ball"].unsqueeze(0)
        left_team_state = state_dict["left_team"].unsqueeze(0)
        left_closest_state = state_dict["left_closest"].unsqueeze(0)
        right_team_state = state_dict["right_team"].unsqueeze(0)
        right_closest_state = state_dict["right_closest"].unsqueeze(0)
        avail = state_dict["avail"].unsqueeze(0)

        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        left_team_embed = self.norm_left(self.fc_left(left_team_state))  # horizon, batch, n, dim
        left_closest_embed = self.norm_left_closest(self.fc_left_closest(left_closest_state))
        right_team_embed = self.norm_right(self.fc_right(right_team_state))
        right_closest_embed = self.norm_right_closest(self.fc_right_closest(right_closest_state))
        [horizon, batch_size, n_player, dim] = left_team_embed.size()
        left_team_embed = left_team_embed.view(horizon * batch_size, n_player,
                                               dim).permute(0, 2, 1)  # horizon * batch, dim1, n
        left_team_embed = F.relu(self.conv1d_left(left_team_embed)).permute(0, 2, 1)  # horizon * batch, n, dim2
        left_team_embed = left_team_embed.reshape(horizon * batch_size,
                                                  -1).view(horizon, batch_size, -1)  # horizon, batch, n * dim2
        left_team_embed = F.relu(self.norm_left2(self.fc_left2(left_team_embed)))

        right_team_embed = right_team_embed.view(horizon * batch_size, n_player + 1,
                                                 dim).permute(0, 2, 1)  # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(0, 2, 1)  # horizon * batch, n * dim2
        ## Usually we need to call reshape() or contiguous() after permute, transpose, etc to make sure
        # tensor on memory is contiguous
        right_team_embed = right_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size, -1)
        ## view() can only be used on contiguous tensor, reshape() don't have this limit.
        right_team_embed = F.relu(self.norm_right2(self.fc_right2(right_team_embed)))

        cat = torch.cat(
            [player_embed, ball_embed, left_team_embed, right_team_embed, left_closest_embed, right_closest_embed], 2
        )
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        hidden = state_dict.pop('prev_state', None)
        if hidden is None:
            h_in = (
                torch.zeros([1, batch_size, self.cfg.lstm_size],
                            dtype=torch.float), torch.zeros([1, batch_size, self.cfg.lstm_size], dtype=torch.float)
            )
        else:
            h_in = hidden
        out, h_out = self.lstm(cat, h_in)

        a_out = F.relu(self.norm_pi_a1(self.fc_pi_a1(out)))
        a_out = self.fc_pi_a2(a_out)
        logit = a_out + (avail - 1) * 1e7
        prob = F.softmax(logit, dim=2)

        v = F.relu(self.norm_v1(self.fc_v1(out)))
        v = self.fc_v2(v)

        return {'logit': prob.squeeze(0), 'value': v.squeeze(0), 'next_state': h_out}

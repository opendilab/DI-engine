import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nervex.utils import MODEL_REGISTRY, deep_merge_dicts
from nervex.config import read_config
from torch.distributions import Categorical
from nervex.model import model_wrap
from nervex.model.actor_critic import FCValueAC
from nervex.torch_utils import to_tensor

conv1d_default_config = read_config(
    './app_zoo/gfootball/model/conv1d/conv1d_default_config.py')


def state_to_tensor(state_dict):
    player_state = torch.from_numpy(
        state_dict["player"]).float().unsqueeze(0).unsqueeze(0)
    ball_state = torch.from_numpy(
        state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)
    left_team_state = torch.from_numpy(
        state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)
    left_closest_state = torch.from_numpy(
        state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(0)
    right_team_state = torch.from_numpy(
        state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)
    right_closest_state = torch.from_numpy(
        state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(0)
    avail = torch.from_numpy(
        state_dict["avail"]).float().unsqueeze(0).unsqueeze(0)
    h = state_dict.pop('hidden', None)
    if h is None:
        h = (torch.zeros([1, 1, conv1d_default_config.LSTM_size], dtype=torch.float), 
                 torch.zeros([1, 1, conv1d_default_config.LSTM_size], dtype=torch.float))

    state_dict_tensor = {
        "player": player_state,
        "ball": ball_state,
        "left_team": left_team_state,
        "left_closest": left_closest_state,
        "right_team": right_team_state,
        "right_closest": right_closest_state,
        "avail": avail,
        "hidden" : h
    }
    return state_dict_tensor


# @MODEL_REGISTRY.register('Conv1D')
# class GfootballConv1DModel(nn.Module):
#     def __init__(
#         self,
#         cfg: dict = {},
#     ) -> None:
#         super(GfootballConv1DModel, self).__init__()
#         cfg_ = conv1d_default_config
#         self.pre_model = model_wrap(
#             GfootballFeatureEmb(cfg), wrapper_name='hidden_state', state_num=256, save_prev_state=False)
#         self.full_model = FCValueAC(obs_dim=cfg_.Policy_Head.input_dim, action_dim=cfg_.Policy_Head.act_shape,
#                                 embedding_dim=cfg_.Policy_Head.input_dim, head_hidden_dim=cfg_.Policy_Head.hidden_dim)
#     def forward(self, data, **kwargs):
#         output = self.pre_model(data, **kwargs)
#         output = self.full_model.forward(output, mode='compute_actor_critic',**kwargs)
#         return output
# class GfootballLSTM(nn.Module):
#     def __init__(
#             self,
#             cfg: dict = {},
#     ) -> None:
#         super(GfootballLSTM, self).__init__()
#         self.model = GfootballFeatureEmb()
#         self._model = HiddenStateWrapper(
#             GfootballFeatureEmb(cfg), state_num=256, save_prev_state=False)
#     def forward(self, data, **kwargs):
#         output = self.model(data)
#         print(f'output:{output}')
#         output = self._model.forward(output, **kwargs)
#         return output
@MODEL_REGISTRY.register('Conv1D')
class GfootballConv1DModel(nn.Module):
    def __init__(
        self,
        cfg: dict = {},
    ) -> None:
        super(GfootballConv1DModel, self).__init__()
        self.device = None
        self.cfg = deep_merge_dicts(conv1d_default_config, cfg)

        self.fc_player = nn.Linear(
            self.cfg.Feature_Enbedding.Player.input_dim, self.cfg.Feature_Enbedding.Player.output_dim)
        self.fc_ball = nn.Linear(
            self.cfg.Feature_Enbedding.Ball.input_dim, self.cfg.Feature_Enbedding.Ball.output_dim)
        self.fc_left = nn.Linear(self.cfg.Feature_Enbedding.LeftTeam.input_dim,
                                 self.cfg.Feature_Enbedding.LeftTeam.output_dim)
        self.fc_right = nn.Linear(self.cfg.Feature_Enbedding.RightTeam.input_dim,
                                  self.cfg.Feature_Enbedding.RightTeam.output_dim)
        self.fc_left_closest = nn.Linear(
            self.cfg.Feature_Enbedding.LeftClosest.input_dim, self.cfg.Feature_Enbedding.LeftClosest.output_dim)
        self.fc_right_closest = nn.Linear(
            self.cfg.Feature_Enbedding.RightClosest.input_dim, self.cfg.Feature_Enbedding.RightClosest.output_dim)

        self.conv1d_left = nn.Conv1d(self.cfg.Feature_Enbedding.LeftTeam.output_dim,
                                     self.cfg.Feature_Enbedding.LeftTeam.conv1d_output_channel, 1, stride=1)
        self.conv1d_right = nn.Conv1d(self.cfg.Feature_Enbedding.RightTeam.output_dim,
                                      self.cfg.Feature_Enbedding.RightTeam.conv1d_output_channel, 1, stride=1)
        self.fc_left2 = nn.Linear(self.cfg.Feature_Enbedding.LeftTeam.conv1d_output_channel *
                                  10, self.cfg.Feature_Enbedding.LeftTeam.fc_output_dim)
        self.fc_right2 = nn.Linear(self.cfg.Feature_Enbedding.RightTeam.conv1d_output_channel *
                                   11, self.cfg.Feature_Enbedding.RightTeam.fc_output_dim)
        self.fc_cat = nn.Linear(self.cfg.FC_CAT.input_dim, self.cfg.LSTM_size)

        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(48)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_left_closest = nn.LayerNorm(48)
        self.norm_right = nn.LayerNorm(48)
        self.norm_right2 = nn.LayerNorm(96)
        self.norm_right_closest = nn.LayerNorm(48)
        self.norm_cat = nn.LayerNorm(self.cfg.LSTM_size)

        self.lstm = nn.LSTM(self.cfg.LSTM_size, self.cfg.LSTM_size)

        self.fc_pi_a1 = nn.Linear(
            self.cfg.LSTM_size, self.cfg.Policy_Head.hidden_dim)
        self.fc_pi_a2 = nn.Linear(
            self.cfg.Policy_Head.hidden_dim, self.cfg.Policy_Head.act_shape)
        self.norm_pi_a1 = nn.LayerNorm(164)

        self.fc_pi_m1 = nn.Linear(self.cfg.LSTM_size, 164)
        self.fc_pi_m2 = nn.Linear(164, 8)
        self.norm_pi_m1 = nn.LayerNorm(164)

        self.fc_v1 = nn.Linear(
            self.cfg.LSTM_size, self.cfg.Value_Head.hidden_dim)
        self.norm_v1 = nn.LayerNorm(164)
        self.fc_v2 = nn.Linear(
            self.cfg.Value_Head.hidden_dim, self.cfg.Value_Head.output_dim,  bias=False)
        # self.optimizer = optim.Adam(
        #     self.parameters(), lr=self.cfg.lr)

    def forward(self, state_dict):
        player_state = state_dict["player"]
        ball_state = state_dict["ball"]
        left_team_state = state_dict["left_team"]
        left_closest_state = state_dict["left_closest"]
        right_team_state = state_dict["right_team"]
        right_closest_state = state_dict["right_closest"]
        avail = state_dict["avail"]

        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        left_team_embed = self.norm_left(self.fc_left(
            left_team_state))  # horizon, batch, n, dim
        left_closest_embed = self.norm_left_closest(
            self.fc_left_closest(left_closest_state))
        right_team_embed = self.norm_right(self.fc_right(right_team_state))
        right_closest_embed = self.norm_right_closest(
            self.fc_right_closest(right_closest_state))

        [horizon, batch_size, n_player, dim] = left_team_embed.size()
        left_team_embed = left_team_embed.view(
            horizon*batch_size, n_player, dim).permute(0, 2, 1)         # horizon * batch, dim1, n
        left_team_embed = F.relu(self.conv1d_left(left_team_embed)).permute(
            0, 2, 1)                       # horizon * batch, n, dim2
        left_team_embed = left_team_embed.reshape(
            horizon*batch_size, -1).view(horizon, batch_size, -1)    # horizon, batch, n * dim2
        left_team_embed = F.relu(self.norm_left2(
            self.fc_left2(left_team_embed)))

        right_team_embed = right_team_embed.view(
            horizon*batch_size, n_player+1, dim).permute(0, 2, 1)    # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(
            0, 2, 1)                   # horizon * batch, n * dim2
        right_team_embed = right_team_embed.reshape(
            horizon*batch_size, -1).view(horizon, batch_size, -1)
        right_team_embed = F.relu(self.norm_right2(
            self.fc_right2(right_team_embed)))

        cat = torch.cat([player_embed, ball_embed, left_team_embed,
                        right_team_embed, left_closest_embed, right_closest_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        h_in = state_dict["hidden"]
        out, h_out = self.lstm(cat, h_in)

        a_out = F.relu(self.norm_pi_a1(self.fc_pi_a1(out)))
        a_out = self.fc_pi_a2(a_out)
        logit = a_out + (avail-1)*1e7
        prob = F.softmax(logit, dim=2)

        v = F.relu(self.norm_v1(self.fc_v1(out)))
        v = self.fc_v2(v)

        return {'action_prob': prob,
                'v': v,
                'hidden': h_out}

    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        s_player_batch, s_ball_batch, s_left_batch, s_left_closest_batch, s_right_batch, s_right_closest_batch, avail_batch = [], [], [], [], [], [], []
        s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch, s_left_closest_prime_batch, \
            s_right_prime_batch, s_right_closest_prime_batch, avail_prime_batch = [
            ], [], [], [], [], [], []
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, m_batch, r_batch, prob_batch, done_batch, need_move_batch = [], [], [], [], [], []

        for rollout in data:
            s_player_lst, s_ball_lst, s_left_lst, s_left_closest_lst, s_right_lst, s_right_closest_lst, avail_lst = [
            ], [], [], [], [], [], []
            s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, s_left_closest_prime_lst, \
                s_right_prime_lst, s_right_closest_prime_lst, avail_prime_lst = [], [], [], [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            a_lst, m_lst, r_lst, prob_lst, done_lst, need_move_lst = [], [], [], [], [], []

            for transition in rollout:
                s, a, m, r, s_prime, prob, done, need_move = transition

                s_player_lst.append(s["player"])
                s_ball_lst.append(s["ball"])
                s_left_lst.append(s["left_team"])
                s_left_closest_lst.append(s["left_closest"])
                s_right_lst.append(s["right_team"])
                s_right_closest_lst.append(s["right_closest"])
                avail_lst.append(s["avail"])
                h1_in, h2_in = s["hidden"]
                h1_in_lst.append(h1_in)
                h2_in_lst.append(h2_in)

                s_player_prime_lst.append(s_prime["player"])
                s_ball_prime_lst.append(s_prime["ball"])
                s_left_prime_lst.append(s_prime["left_team"])
                s_left_closest_prime_lst.append(s_prime["left_closest"])
                s_right_prime_lst.append(s_prime["right_team"])
                s_right_closest_prime_lst.append(s_prime["right_closest"])
                avail_prime_lst.append(s_prime["avail"])
                h1_out, h2_out = s_prime["hidden"]
                h1_out_lst.append(h1_out)
                h2_out_lst.append(h2_out)

                a_lst.append([a])
                m_lst.append([m])
                r_lst.append([r])
                prob_lst.append([prob])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                need_move_lst.append([need_move]),

            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_left_batch.append(s_left_lst)
            s_left_closest_batch.append(s_left_closest_lst)
            s_right_batch.append(s_right_lst)
            s_right_closest_batch.append(s_right_closest_lst)
            avail_batch.append(avail_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])

            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_left_closest_prime_batch.append(s_left_closest_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)
            s_right_closest_prime_batch.append(s_right_closest_prime_lst)
            avail_prime_batch.append(avail_prime_lst)
            h1_out_batch.append(h1_out_lst[0])
            h2_out_batch.append(h2_out_lst[0])

            a_batch.append(a_lst)
            m_batch.append(m_lst)
            r_batch.append(r_lst)
            prob_batch.append(prob_lst)
            done_batch.append(done_lst)
            need_move_batch.append(need_move_lst)

        s = {
            "player": torch.tensor(s_player_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "right_team": torch.tensor(s_right_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "avail": torch.tensor(avail_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2))
        }

        s_prime = {
            "player": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "right_team": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "avail": torch.tensor(avail_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2))
        }

        a, m, r, done_mask, prob, need_move = torch.tensor(a_batch, device=self.device).permute(1, 0, 2), \
            torch.tensor(m_batch, device=self.device).permute(1, 0, 2), \
            torch.tensor(r_batch, dtype=torch.float, device=self.device).permute(1, 0, 2), \
            torch.tensor(done_batch, dtype=torch.float, device=self.device).permute(1, 0, 2), \
            torch.tensor(prob_batch, dtype=torch.float, device=self.device).permute(1, 0, 2), \
            torch.tensor(need_move_batch, dtype=torch.float,
                         device=self.device).permute(1, 0, 2)

        return s, a, m, r, s_prime, done_mask, prob, need_move

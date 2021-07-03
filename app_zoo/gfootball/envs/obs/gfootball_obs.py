import numpy as np
import torch
import math

from ding.envs.common import EnvElement
from functools import partial
from ding.torch_utils import one_hot
from ding.envs.common import div_func, div_one_hot

N_PLAYER = 11


def score_preprocess(scores):
    ret = []
    for score in scores:
        clip_score = torch.clamp_max(score.unsqueeze(0), 10)  # 0-9: 0-9; 10: >=10
        ret.append(one_hot(clip_score, num=11).squeeze(0))
    return torch.cat(ret, dim=0)


class MatchObs(EnvElement):
    _name = "GFootballMatchObs"

    def _init(self, cfg):
        self._default_val = None
        self.template = [
            # ------Ball information
            {
                'key': 'ball',
                'ret_key': 'ball_position',
                'dim': 3,
                'op': lambda x: x,
                'value': {
                    'min': (-1, -0.42, 0),
                    'max': (1, 0.42, 100),
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float (x, y, z)'
            },
            {
                'key': 'ball_direction',
                'ret_key': 'ball_direction',
                'dim': 3,
                'op': lambda x: x,
                'value': {
                    'min': (-1, -0.42, 0),
                    'max': (1, 0.42, 100),
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float (x, y, z)'
            },
            {
                'key': 'ball_rotation',
                'ret_key': 'ball_rotation',
                'dim': 3,
                'op': lambda x: x,
                'value': {
                    'min': (-math.pi, -math.pi, -math.pi),
                    'max': (math.pi, math.pi, math.pi),
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float (x, y, z)'
            },
            {
                'key': 'ball_owned_team',
                'ret_key': 'ball_owned_team',
                'dim': 3,
                'op': lambda x: partial(one_hot, num=3)(x + 1),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one hot 3 value',
                'meaning': ['NotOwned', 'LeftTeam', 'RightTeam']
            },
            {
                'key': 'ball_owned_player',
                'ret_key': 'ball_owned_player',
                'dim': N_PLAYER + 1,  # 0...N_1: player_idx, N: nobody
                'op': lambda x: partial(one_hot, num=N_PLAYER + 1)(x + N_PLAYER + 1 if x == -1 else x),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one hot 12 value',
                'meaning': 'index of player'
            },
            # ------Controlled player information
            {
                'key': 'active',
                'ret_key': 'active_player',
                'dim': N_PLAYER,
                'op': partial(one_hot, num=N_PLAYER),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one hot 11 value',
                'meaning': 'index of controlled player'
            },
            {
                'key': 'designated',  # In non-multiagent mode it is always equal to `active`
                'ret_key': 'designated_player',
                'dim': N_PLAYER,
                'op': partial(one_hot, num=N_PLAYER),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one hot 11 value',
                'meaning': 'index of player'
            },
            {
                'key': 'sticky_actions',
                'ret_key': 'active_player_sticky_actions',
                'dim': 10,
                'op': lambda x: x,
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'boolean vector'
                },
                'other': 'boolean vector with 10 value',
                'meaning': [
                    'Left', 'TopLeft', 'Top', 'TopRight', 'Right', 'BottomRight', 'Bottom', 'BottomLeft', 'Sprint',
                    'Dribble'
                ]  # 8 directions are one-hot
            },
            # ------Match state
            {
                'key': 'score',
                'ret_key': 'score',
                'dim': 22,
                'op': score_preprocess,
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'each score one hot 11 values(10 for 0-9, 1 for over 10), concat two scores',
            },
            {
                'key': 'steps_left',
                'ret_key': 'steps_left',
                'dim': 30,
                'op': partial(div_one_hot, max_val=2999, ratio=100),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'div(50), one hot 30 values',
            },
            {
                'key': 'game_mode',
                'ret_key': 'game_mode',
                'dim': 7,
                'op': partial(one_hot, num=7),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one-hot 7 values',
                'meaning': ['Normal', 'KickOff', 'GoalKick', 'FreeKick', 'Corner', 'ThrowIn', 'Penalty']
            },
        ]
        self.cfg = cfg
        self._shape = {t['key']: t['dim'] for t in self.template}
        self._value = {t['key']: t['value'] for t in self.template}
        self._to_agent_processor = self.parse
        self._from_agent_processor = None

    def parse(self, obs: dict) -> dict:
        '''
            Overview: find corresponding setting in cfg, parse the feature
            Arguments:
                - feature (:obj:`ndarray`): the feature to parse
                - idx_dict (:obj:`dict`): feature index dict
            Returns:
                - ret (:obj:`list`): parse result tensor list
        '''
        ret = {}
        for item in self.template:
            key = item['key']
            ret_key = item['ret_key']
            data = obs[key]
            if not isinstance(data, list):
                data = [data]
            data = torch.Tensor(data) if item['value']['dinfo'] != 'one-hot' else torch.LongTensor(data)
            try:
                data = item['op'](data)
            except RuntimeError:
                print(item, data)
                raise RuntimeError
            if len(data.shape) == 2:
                data = data.squeeze(0)
            ret[ret_key] = data.numpy()
        return ret

    def _details(self):
        return 'Match Global Obs: Ball, Controlled Player and Match State'


class PlayerObs(EnvElement):
    _name = "GFootballPlayerObs"

    def _init(self, cfg):
        self._default_val = None
        self.template = [
            {
                'key': 'team',
                'ret_key': 'team',
                'dim': 2,
                'op': partial(one_hot, num=2),  # 0 for left, 1 for right
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one-hot 2 values for which team'
            },
            {
                'key': 'index',
                'ret_key': 'index',
                'dim': N_PLAYER,
                'op': partial(one_hot, num=N_PLAYER),
                'value': {
                    'min': 0,
                    'max': N_PLAYER,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one-hot N_PLAYER values for index in one team'
            },
            {
                'key': 'position',
                'ret_key': 'position',
                'dim': 2,
                'op': lambda x: x,
                'value': {
                    'min': (-1, -0.42),
                    'max': (1, 0.42),
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float (x, y)'
            },
            {
                'key': 'direction',
                'ret_key': 'direction',
                'dim': 2,
                'op': lambda x: x,
                'value': {
                    'min': (-1, -0.42),
                    'max': (1, 0.42),
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float'
            },
            {
                'key': 'tired_factor',
                'ret_key': 'tired_factor',
                'dim': 1,
                'op': lambda x: x,
                'value': {
                    'min': (0, ),
                    'max': (1, ),
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float'
            },
            {
                'key': 'yellow_card',
                'ret_key': 'yellow_card',
                'dim': 2,
                'op': partial(one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one hot 2 values'
            },
            {
                'key': 'active',  # 0(False) means got a red card
                'ret_key': 'active',
                'dim': 2,
                'op': partial(one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'float'
            },
            {
                'key': 'roles',
                'ret_key': 'role',
                'dim': 10,
                'op': partial(one_hot, num=10),
                'value': {
                    'min': 0,
                    'max': 2,
                    'dtype': float,
                    'dinfo': 'one-hot'
                },
                'other': 'one-hot 10 values',
                'meaning': [
                    'GoalKeeper', 'CentreBack', 'LeftBack', 'RightBack', 'DefenceMidfield', 'CentralMidfield',
                    'LeftMidfield', 'RightMidfield', 'AttackMidfield', 'CentralFront'
                ]
            },
        ]
        self.cfg = cfg
        self._shape = {'players': {t['key']: t['dim'] for t in self.template}}
        self._value = {'players': {t['key']: t['value'] for t in self.template}}
        self._to_agent_processor = self.parse
        self._from_agent_processor = None

    def parse(self, obs: dict) -> dict:
        players = []
        for player_idx in range(N_PLAYER):
            players.append(self._parse(obs, 'left_team', player_idx))
        for player_idx in range(N_PLAYER):
            players.append(self._parse(obs, 'right_team', player_idx))
        return {'players': players}

    def _parse(self, obs: dict, left_right: str, player_idx) -> dict:
        player_dict = {
            'team': 0 if left_right == 'left_team' else 1,
            'index': player_idx,
        }
        for item in self.template:
            key = item['key']
            ret_key = item['ret_key']
            if key in ['team', 'index']:
                data = player_dict[key]
            elif key == 'position':
                player_stat = left_right
                data = obs[player_stat][player_idx]
            else:
                player_stat = left_right + '_' + key
                data = obs[player_stat][player_idx]
            if not isinstance(data, np.ndarray):
                data = [data]
            data = torch.Tensor(data) if item['value']['dinfo'] != 'one-hot' else torch.LongTensor(data)
            try:
                data = item['op'](data)
            except RuntimeError:
                print(item, data)
                raise RuntimeError
            if len(data.shape) == 2:
                data = data.squeeze(0)
            player_dict[ret_key] = data.numpy()
        return player_dict

    def _details(self):
        return 'Single Player Obs'


class FullObs(EnvElement):
    _name = "GFootballFullObs"

    def _init(self, cfg):
        self._default_val = None
        self.template = [
            {
                'key': 'player',
                'ret_key': 'player',
                'dim': 36,
                'op': lambda x: x,
                'value': {
                    'min': (
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -0.42, -1, -0.42, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0
                    ),
                    'max': (
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.42, 1, 0.42, float(np.inf), 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                    ),
                    'dtype': float,
                    'dinfo': 'mix'
                },
                'other': 'mixed active player info'
            },
            {
                'key': 'ball',
                'ret_key': 'ball',
                'dim': 18,
                'op': lambda x: x,
                'value': {
                    'min': (-1, -0.42, 0, 0, 0, 0, 0, 0, 0, -2, -0.84, -20, -8.4, 0, 0, 0, 0, 0),
                    'max': (1, 0.42, 100, 1, 1, 1, 1, 1, 1, 2, 0.84, 20, 8.4, np.inf, np.inf, 2.5, 1, 1),
                    'dtype': float,
                    'dinfo': 'mix'
                },
                'other': 'mixed ball info, relative to active player'
            },
            {
                'key': 'LeftTeam',
                'ret_key': 'LeftTeam',
                'dim': 7,
                'op': lambda x: x,
                'value': {
                    'min': (-1, -0.42, -1, -0.42, 0, 0, 0),
                    'max': (1, 0.42, 1, 0.42, 100, 2.5, 1),
                    'dtype': float,
                    'dinfo': 'mix'
                },
                'other': 'mixed player info, relative to active player,\
                 will have 10+1 infos(all left team member and closest member )'
            },
            {
                'key': 'RightTeam',
                'ret_key': 'RightTeam',
                'dim': 7,
                'op': lambda x: x,
                'value': {
                    'min': (-1, -0.42, -1, -0.42, 0, 0, 0),
                    'max': (1, 0.42, 1, 0.42, 100, 2.5, 1),
                    'dtype': float,
                    'dinfo': 'mix'
                },
                'other': 'mixed player info, relative to active player,\
                 will have 10+1 infos(all right team member and closest member )'
            },
        ]
        self.cfg = cfg
        self._shape = {t['key']: t['dim'] for t in self.template}
        self._value = {t['key']: t['value'] for t in self.template}

    def _details(self):
        return 'Full Obs for Gfootball Self Play'

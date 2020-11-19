import numpy as np
from typing import Union

from .player import Player


class Payoff:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """
    data_keys = ['wins', 'draws', 'losses', 'games']

    def __init__(self, home_id: str, decay: float, min_win_rate_games: int = 8) -> None:
        """
        Overview: Initialize payoff
        Arguments:
            - home_id (:obj:`str`): home player id
            - decay (:obj:`float`): update step decay factor
            - min_win_rate_games (:obj:`int`): min games for win rate calculation
        """
        # self._players is a list including the reference(shallow copy) of all the possible opponent player
        self._players = []
        # self._data is a dict, whose key is the player_id of the element of self._players,
        # and whose value is a dict with keys like `data_keys`
        self._data = {}
        self._home_id = home_id
        self._decay = decay
        self._min_win_rate_games = min_win_rate_games

    def __getitem__(self, players: Union[list, Player]) -> np.array:
        """
        Overview:
            Get win rates for a players list or a player
        Arguments:
            - players (:obj:`list or Player`): a player or a player list to access win rates
        Returns:
            - win_rates (:obj:`np.array`): win rate numpy array
        """
        assert isinstance(players, list) or isinstance(players, Player)
        # single player case
        if isinstance(players, Player):
            players = [players]
        return np.array([self._win_rate(p) for p in players])

    def _win_rate(self, player: Player) -> float:
        """
        Overview:
            Get win rate against an opponent player
        Arguments:
            - player (:obj:`Player`): the opponent player to calculate win rate
        Returns:
            - win rate (:obj:`float`): float win rate value. \
                Only when total games is no less than ``self._min_win_rate_games``, \
                can the win rate be calculated according to [win, draw, loss, game], or return 0.5 by default.
        """
        key = player.player_id
        handle = self._data[key]
        # not enough game record case
        if handle['games'] < self._min_win_rate_games:
            return 0.5

        return (handle['wins'] + 0.5 * handle['draws']) / (handle['games'])

    @property
    def players(self) -> list:
        return self._players

    def add_player(self, player: Player) -> None:
        """
        Overview:
            Add a player to ``self._players`` and update ``self._data`` for corresponding initialization
        Arguments:
            - player (:obj:`Player`): the player to be added
        """
        self._players.append(player)
        key = player.player_id
        self._data[key] = {k: 0 for k in self.data_keys}

    def update(self, match_info: dict) -> bool:
        """
        Overview:
            Update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information,
                owning at least 3 keys('home_id', 'away_id', 'result')
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        # check
        try:
            assert match_info['home_id'] == self._home_id
            assert match_info['away_id'] in self._data.keys()
            assert match_info['result'] in self.data_keys[:3]
        except Exception:
            print("[ERROR] invalid match_info: {}".format(match_info))
            return False
        # decay
        key = match_info['away_id']
        for k in self.data_keys:
            self._data[key][k] *= self._decay

        # update
        self._data[key]['games'] += 1
        self._data[key][match_info['result']] += 1
        return True

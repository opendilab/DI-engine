import numpy as np
from .player import Player


class Payoff:
    """
    Overview: payoff data structure to record historical match result, each player owns one specific payoff
    Interface: __init__, __getitem__, add_player, update
    Property: players
    """
    data_keys = ['wins', 'draws', 'losses', 'games']

    def __init__(self, home_id, decay, min_win_rate_games=8):
        """
        Overview: initialize payoff
        Arguments:
            - home_id (:obj:`str`): home player id
            - decay (:obj:`float`): update step decay factor
            - min_win_rate_games (:obj:`int`): min games for win rate calculation
        """
        # self._players is a list including the reference(shallow copy) of all the possible opponent player
        self._players = []
        # self._data is a dict, whose key is the player_id of the element of self._players,
        # and whose value is a dict about the attributes in self.data_keys
        self._data = {}
        self._home_id = home_id
        self._decay = decay
        self._min_win_rate_games = min_win_rate_games

    def __getitem__(self, players):
        """
        Overview: get win rates for a players list or a player
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

    def _win_rate(self, player):
        """
        Overview: get win rate for a player
        Arguments:
            - player (:obj:`Player`): a player to access win rate
        Returns:
            - win rate (:obj:`float`): float win rate value
        """
        key = player.player_id
        handle = self._data[key]
        # no game record case
        if handle['games'] < self._min_win_rate_games:
            return 0.5

        return (handle['wins'] + 0.5 * handle['draws']) / (handle['games'])

    @property
    def players(self):
        """
        Overview: get all the players
        Returns:
            - players (:obj:`list`): players list
        """
        return self._players

    def add_player(self, player):
        """
        Overview: add a player and do the corresponding initialization
        Arguments:
            - player (:obj:`Player`): a player to be added
        """
        self._players.append(player)
        key = player.player_id
        self._data[key] = {k: 0 for k in self.data_keys}

    def update(self, match_info):
        """
        Overview: update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information
        Returns:
            - result (:obj:`bool`): whether update is successful
        Note:
            match_info owns at least 3 keys('home', 'away', 'result')
        """
        # check
        try:
            assert match_info['home'] == self._home_id
            assert match_info['away'] in self._data.keys()
            assert match_info['result'] in self.data_keys[:3]
        except Exception:
            print("[ERROR] invalid match_info: {}".format(match_info))
            return False
        # decay
        key = match_info['away']
        for k in self.data_keys:
            self._data[key][k] *= self._decay

        # update
        self._data[key]['games'] += 1
        self._data[key][match_info['result']] += 1
        return True

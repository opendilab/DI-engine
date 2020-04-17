import numpy as np
import copy
from collections import defaultdict
from .player import Player
from sc2learner.utils import LockContext


class RecordDict(dict):
    data_keys = ['wins', 'draws', 'losses', 'games']

    def __init__(self):
        super(RecordDict, self).__init__()
        for k in self.data_keys:
            self[k] = 0

    def __mul__(self, val):
        obj = copy.deepcopy(self)
        for k in obj.keys():
            obj[k] *= val
        return obj


class PayoffDict(defaultdict):
    def __init__(self, init_fn=None):
        if init_fn is None:
            init_fn = RecordDict
        super(PayoffDict, self).__init__(init_fn)


class SharedPayoff:
    """
    Overview: payoff data structure to record historical match result, this payoff is shared among all the players
    Interface: __init__, __getitem__, add_player, update, get_key
    Property: players
    """

    # TODO(nyz) whether ensures the thread-safe

    def __init__(self, decay, min_win_rate_games=8):
        """
        Overview: initialize payoff
        Arguments:
            - decay (:obj:`float`): update step decay factor
        """
        # self._players is a list including the reference(shallow copy) of all the possible player
        self._players = []
        self._players_ids = []
        # self._data is a PayoffDict, whose key is the player_id of the element of self._players,
        # and whose value is a RecordDict
        self._data = PayoffDict()
        self._decay = decay
        self._min_win_rate_games = min_win_rate_games
        self._lock = LockContext(lock_type='thread')

    def __getitem__(self, players):
        """
        Overview: get win rates for a players list or a player
        Arguments:
            - players (:obj:`tuple`): a tuple(home, away), each one is a player or a player list to access win rates
        Returns:
            - win_rates (:obj:`np.array`): win rate numpy array
        """
        with self._lock:
            home, away = players
            assert isinstance(home, list) or isinstance(home, Player)
            assert isinstance(away, list) or isinstance(away, Player)
            # single player case
            if isinstance(home, Player):
                home = [home]
            if isinstance(away, Player):
                away = [away]

            win_rates = np.array([[self._win_rate(h.player_id, a.player_id) for a in away] for h in home])
            if len(home) == 1 or len(away) == 1:
                win_rates = win_rates.reshape(-1)

            return win_rates

    def _win_rate(self, home, away):
        """
        Overview: get win rate for home player VS away player
        Arguments:
            - home (:obj:`Player`): the home player to access win rate
            - away (:obj:`Player`): the away player to access win rate
        Returns:
            - win rate (:obj:`float`): float win rate value
        """
        key = self.get_key(home, away)
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
        with self._lock:
            return self._players

    def add_player(self, player):
        """
        Overview: add a player
        Arguments:
            - player (:obj:`Player`): a player to be added
        """
        with self._lock:
            self._players.append(player)
            self._players_ids.append(player.player_id)

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
        with self._lock:
            try:
                assert match_info['home_id'] in self._players_ids
                assert match_info['away_id'] in self._players_ids
                assert match_info['result'] in RecordDict.data_keys[:3]
            except Exception as e:
                print("[ERROR] invalid match_info: {}".format(match_info))
                print(e)
                return False
            # decay
            key = self.get_key(match_info['home_id'], match_info['away_id'])
            self._data[key] *= self._decay

            # update
            self._data[key]['games'] += 1
            self._data[key][match_info['result']] += 1
            return True

    def get_key(self, home, away):
        assert isinstance(home, str)
        assert isinstance(away, str)
        if home <= away:
            tmp = [home, away]
        else:
            tmp = [away, home]
        return '-'.join(tmp)

import copy
from collections import defaultdict, deque
from typing import Union, Tuple
from functools import partial
from easydict import EasyDict

import numpy as np
from nervex.utils import LockContext, LockContextType

from .player import Player


class BattleRecordDict(dict):
    """
    Overview:
        A dict used to record battle game result.
        Initialized with four fixed keys: ['wins', 'draws', 'losses', 'games'] with value 0
    Interfaces:
        __init__, __mul__
    """
    data_keys = ['wins', 'draws', 'losses', 'games']

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialize four fixed keys ['wins', 'draws', 'losses', 'games'] and set value to 0
        """
        super(BattleRecordDict, self).__init__()
        for k in self.data_keys:
            self[k] = 0

    def __mul__(self, val: float) -> dict:
        """
        Overview:
            Multiply each key's value with the input multiplier `val`
        Arguments:
            - val (:obj:`float`): the multiplier
        Returns:
            - obj (:obj:`dict`): a deepcopied RecordDict after multiplication
        """
        obj = copy.deepcopy(self)
        for k in obj.keys():
            obj[k] *= val
        return obj


class SoloRecordQueue(deque):
    """
    Overview:
        A deque used to record solo game result. Initialized with maxlen.
    Interfaces:
        __init__, __mul__
    """
    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialize solo game's record queue at ``maxlen`` of ``cfg.buffer_size``
        """
        super(SoloRecordQueue, self).__init__(maxlen=cfg.buffer_size)


class PayoffDict(defaultdict):
    """
    Overview:
        Payoff defaultdict.
        If key doesn't exist, return a data structure instance
        (now supports ``BattleRecordDict``, ``SoloRecordQueue``) set in advance.
    Interfaces:
        __init__, __missing__
    """

    def __init__(self, init_fn: type, cfg: EasyDict):
        """
        Overview:
            Init method, set defaultdict's default return instance type as ``init_fn``.
        Arguments:
            - init_fn (:obj:`type`): if key is missing, PayoffDict can return the instance `init_fn()`
            - cfg (:obj:`EasyDict`): for SoloRecordQueue, containing {buffer_size}
        """
        super(PayoffDict, self).__init__(partial(init_fn, cfg=cfg))


class BattleSharedPayoff:
    """
    Overview:
        Payoff data structure to record historical match result, this payoff is shared among all the players.
        Use LockContext to ensure thread safe, since all players from all threads can access and modify it.
    Interface:
        __init__, __getitem__, add_player, update, get_key
    Property:
        players
    """

    # TODO(nyz) whether ensures the thread-safe

    def __init__(self, cfg: EasyDict):
        """
        Overview: initialize battle payoff
        Arguments:
            - cfg (:obj:`dict`): config(contains {decay, min_win_rate_games})
        """
        # self._players is a list including the reference(shallow copy) of all players,
        # while self._players_ids is a list of string
        self._players = []
        self._players_ids = []
        # self._data is a PayoffDict, whose key is '[player_id]-[player_id]' string,
        # and whose value is a RecordDict
        self._data = PayoffDict(BattleRecordDict, cfg)

        # self._decay controls how past game info (win, draw, loss) decays
        self._decay = cfg.decay
        # self._min_win_rate_games is used in self._win_rate's calculating win_rate
        self._min_win_rate_games = cfg.get('min_win_rate_games', 8)
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)

    def __getitem__(self, players: tuple) -> np.array:
        """
        Overview:
            Get win rates between home players and away players one by one
        Arguments:
            - players (:obj:`tuple`): a tuple(home, away), each one is a player or a players list
        Returns:
            - win_rates (:obj:`np.array`): win rate (squeezed, see Shape for more details) \
                between each player from home and each player from away
        Shape:
            - win_rates: Assume there are m home players and n away players.

                - m != 0 and n != 0: shape is (m, n)
                - m == 0: shape is (n)
                - n == 0: shape is (m)
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

    def _win_rate(self, home: str, away: str) -> float:
        """
        Overview:
            Get win rate of one `home player` vs one `away player`
        Arguments:
            - home (:obj:`str`): home player id to access win rate
            - away (:obj:`str`): away player id to access win rate
        Returns:
            - win rate (:obj:`float`): float win rate value. \
                Only when total games is no less than ``self._min_win_rate_games``, \
                can the win rate be calculated according to [win, draw, loss, game], or return 0.5 by default.
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

    def add_player(self, player: Player) -> None:
        """
        Overview: Add a player
        Arguments:
            - player (:obj:`Player`): the player to be added
        """
        with self._lock:
            self._players.append(player)
            self._players_ids.append(player.player_id)

    def update(self, job_info: dict) -> bool:
        """
        Overview:
            Update payoff with job_info when a job is to be finished.
            If update succeeds, return True; If raises an exception, return False.
        Arguments:
            - job_info (:obj:`dict`): a dict containing job result information
        Returns:
            - result (:obj:`bool`): whether update is successful
        Note:
            job_info owns at least 5 keys('launch_player', 'player_id', 'env_num', 'episode_num', 'result')
            player_id's value is a tuple (home_id, away_id)
            result's value is a two-layer list with the length of (episode_num, env_num)
        """
        def _win_loss_reverse(result_: str, reverse_: bool) -> str:
            if result_ == 'draws' or not reverse_:
                return result_
            reverse_dict = {'wins': 'losses', 'losses': 'wins'}
            return reverse_dict[result_]

        # check
        with self._lock:
            home_id, away_id = job_info['player_id']
            try:
                assert home_id in self._players_ids
                assert away_id in self._players_ids
                assert all([i in BattleRecordDict.data_keys[:3] for j in job_info['result'] for i in j])
            except Exception as e:
                print("[ERROR] invalid job_info: {}".format(job_info))
                print(e)
                return False
            key, reverse = self.get_key(home_id, away_id)
            # update (including decay)
            for j in job_info['result']:
                for i in j:
                    # all categories should decay
                    self._data[key] *= self._decay
                    self._data[key]['games'] += 1
                    result = _win_loss_reverse(i, reverse)
                    self._data[key][result] += 1
            return True

    def get_key(self, home: str, away: str) -> Tuple[str, bool]:
        """
        Overview: Join home player id and away player id
        Arguments:
            - home (:obj:`str`): home player id
            - away (:obj:`str`): away player id
        Returns:
            - key (:obj:`str`): tow ids sorted in alphabetical order, and joined by '-'
        """
        assert isinstance(home, str)
        assert isinstance(away, str)
        reverse = False
        if home <= away:
            tmp = [home, away]
        else:
            tmp = [away, home]
            reverse = True
        return '-'.join(tmp), reverse


class SoloSharedPayoff:
    """
    Overview:
        Payoff data structure to record historical match result.
        Unlike BattleSharedPayoff record battle game results between players,
        SoloSharedPayoff record solo game results only of one player.
    Interface:
        __init__, add_player, update
    """

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview: initialize solo payoff
        Arguments:
            - cfg (:obj:`dict`): config(contains {buffer_size})
        """
        self._data = PayoffDict(SoloRecordQueue, cfg)

    def __getitem__(self, player_id: str) -> SoloRecordQueue:
        """
        Overview:
            Get game result info of the player
        Arguments:
            - player_id (:obj:`str`): the only active player's id
        Returns:
            - record (:obj:`SoloRecordQueue`): record queue with several game results
        """
        return self._data[player_id]

    def update(self, job_info: dict) -> bool:
        """
        Overview: append job_info at the right end of ``self._data``
        Arguments:
            - job_info (:obj:`dict`): a dict containing job result information
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        # TODO(zlx): job_info validation
        self._data.append(job_info)
        return True

    def add_player(self, player: Player) -> None:
        """
        Overview: Will not add a player, since there is only one player.
        """
        pass


def create_payoff(cfg: EasyDict) -> Union[BattleSharedPayoff, SoloSharedPayoff]:
    """
    Overview:
        Given the key (payoff type), now supports keys ['solo', 'battle'],
        create a new payoff instance if in payoff_mapping's values, or raise an KeyError.
    Arguments:
        - cfg (:obj:`EasyDict`): payoff config containing at least one key 'type'
    Returns:
        - payoff (:obj:`BattleSharedPayoff` or :obj:`SoloSharedPayoff`): the created new payoff, \
            should be an instance of one of payoff_mapping's values
    """
    payoff_mapping = {'solo': SoloSharedPayoff, 'battle': BattleSharedPayoff}
    payoff_type = cfg.type
    if payoff_type not in payoff_mapping.keys():
        raise KeyError("not support payoff type: {}".format(payoff_type))
    else:
        return payoff_mapping[payoff_type](cfg)

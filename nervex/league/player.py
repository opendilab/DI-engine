from typing import Callable, Optional, List, Union, Any
from collections import namedtuple
import numpy as np
from easydict import EasyDict

from nervex.utils import deep_merge_dicts, import_module
from nervex.rl_utils import epsilon_greedy


class Player:
    """
    Overview:
        Base player class, player is the basic member of a league
    Interfaces:
        __init__
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "BasePlayer"  # override this variable for sub-class player

    def __init__(self, cfg: EasyDict, category: str,
                 init_payoff: Union['BattleSharedPayoff', 'SoloSharedPayoff'],  # noqa
                 checkpoint_path: str, player_id: str, total_agent_step: int) -> None:
        """
        Overview:
            Initialize base player metadata
        Arguments:
            - cfg (:obj:`EasyDict`): player config dict
            - category (:obj:`str`): player category, depending on the game, \
                e.g. StarCraft has 3 races ['terran', 'protoss', 'zerg']
            - init_payoff (:obj:`BattleSharedPayoff` or :obj:`SoloSharedPayoff`): payoff shared by all players
            - checkpoint_path (:obj:`str`): one training phase step
            - player_id (:obj:`str`): player id
            - total_agent_step (:obj:`int`):  for active player, it should be 0, \
                for historical player, it should be parent player's ``_total_agent_step`` when ``snapshot``
        """
        self._cfg = cfg
        self._category = category
        self._payoff = init_payoff
        self._checkpoint_path = checkpoint_path
        assert isinstance(player_id, str)
        self._player_id = player_id
        self._total_agent_step = total_agent_step

    @property
    def category(self) -> str:
        return self._category

    @property
    def payoff(self) -> Union['BattleSharedPayoff', 'SoloSharedPayoff']:  # noqa
        return self._payoff

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def player_id(self) -> str:
        return self._player_id

    @property
    def total_agent_step(self) -> int:
        return self._total_agent_step

    @total_agent_step.setter
    def total_agent_step(self, step: int) -> None:
        self._total_agent_step = step


class HistoricalPlayer(Player):
    """
    Overview:
        Historical player with fixed checkpoint, has a unique attribute ``parent_id``.
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step, parent_id
    """
    _name = "HistoricalPlayer"

    def __init__(self, *args, parent_id: str) -> None:
        """
        Overview:
            Initialize ``_parent_id`` additionally
        Arguments:
            - parent_id (:obj:`str`): id of historical player's parent, should be an active player
        """
        super(HistoricalPlayer, self).__init__(*args)
        self._parent_id = parent_id

    @property
    def parent_id(self) -> str:
        return self._parent_id


class ActivePlayer(Player):
    """
    Overview:
        Active player class, active player can be updated
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "ActivePlayer"

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize player metadata, depending on the game
        Note:
            - one_phase_step (:obj:`int`): active player will be considered trained enough after one phase step
            - last_enough_step (:obj:`int`): player's last step number that satisfies ``_is_trained_enough``
            - exploration (:obj:`function`): exploration function, e.g. epsilon greedy with decay
        """
        super(ActivePlayer, self).__init__(*args)
        self._one_phase_step = int(float(self._cfg.one_phase_step))  # ``one_phase_step`` is like 1e9
        self._last_enough_step = 0
        if 'exploration' in self._cfg.forward_kwargs:
            self._exploration = epsilon_greedy(self._cfg.forward_kwargs.exploration.start,
                                               self._cfg.forward_kwargs.exploration.end,
                                               self._cfg.forward_kwargs.exploration.decay_len)
        else:
            self._exploration = None

    def is_trained_enough(self, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        step_passed = self._total_agent_step - self._last_enough_step
        if step_passed < self._one_phase_step:
            return False
        else:
            self._last_enough_step = self._total_agent_step
            return True

    def snapshot(self) -> HistoricalPlayer:
        """
        Overview:
            Generate a snapshot historical player from the current player, called in league manager's ``_snapshot``.
        Returns:
            - snapshot_player (:obj:`HistoricalPlayer`): new instantiated historical player
        Note:
            This method only generates a historical player object without saving the checkpoint, which should be
            completed by the interaction between coordinator and learner.
        """
        path = self.checkpoint_path.split('.pth')[0] + '_{}'.format(self._total_agent_step) + '.pth'
        return HistoricalPlayer(
            self._cfg,
            self.category,
            self.payoff,
            path,
            self.player_id + '_{}'.format(int(self._total_agent_step)),
            self._total_agent_step,
            parent_id=self.player_id
        )

    def mutate(self, info: dict) -> Optional[str]:
        """
        Overview:
            Mutate the current player
        Arguments:
            - info (:obj:`dict`): related information for the mutation
        Returns:
            - mutation_result (:obj:`str`): if the player does the mutation operation then returns the
                corresponding model path, otherwise returns None
        """
        pass

    def get_job(self) -> dict:
        """
        Overview:
            Get a dict containing some info about the job to be launched. The dict contains at least 3 keys
            ['forward_kwargs', 'adder_kwargs', 'env_kwargs']. Calls three corresponding methods ``self._get_job_*``
            to get value of each key.
            Apart from those 3 keys, it can also contain keys like ['agent_update_freq', 'compressor'].
            For battle active player, it should contain the selected opponent.
        Returns:
            - ret (:obj:`dict`): the returned dict, containing at least 3 keys \
                ['forward_kwargs', 'adder_kwargs', 'env_kwargs']
        Note:
            - forward_kwargs: e.g. decayed epsilon value for exploration
            - env_kwargs: e.g. game mode, scenario, difficulty
            - adder_kwargs: e.g. whether to use gae, data push length
        """
        job_dict = self._cfg.job
        return deep_merge_dicts({
            'forward_kwargs': self._get_job_forward(),
            'adder_kwargs': self._get_job_adder(),
            'env_kwargs': self._get_job_env()
        }, job_dict)

    def _get_job_forward(self) -> dict:
        ret = {}
        if 'exploration' in self._cfg.forward_kwargs:
            ret['eps'] = self._exploration(self.total_agent_step)
        return ret

    def _get_job_adder(self) -> dict:
        return self._cfg.adder_kwargs

    def _get_job_env(self) -> dict:
        return self._cfg.env_kwargs


class BattleActivePlayer(ActivePlayer):
    """
    Overview:
        Active player class for battle games
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "BattleActivePlayer"
    BRANCH = namedtuple("BRANCH", ['name', 'prob'])

    # override
    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize league player metadata additionally
        Note:
            - strong_win_rate (:obj:`float`): if win rates between this player and all the opponents are greater than
                this value, this player can be regarded as strong enough to these opponents, therefore trained enough
            - branch_probs (:obj:`namedtuple`): a namedtuple of probabilities of selecting different opponent branch
        """
        super(BattleActivePlayer, self).__init__(*args, **kwargs)
        self._strong_win_rate = self._cfg.strong_win_rate
        assert isinstance(self._cfg.branch_probs, dict)
        self._branch_probs = [self.BRANCH(k, v) for k, v in self._cfg.branch_probs.items()]

    # override
    def is_trained_enough(self, select_fn: Callable) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Arguments:
            - select_fn (:obj:`function`): function to select historical players
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        step_passed = self._total_agent_step - self._last_enough_step
        if step_passed < self._one_phase_step:
            return False
        elif step_passed >= 2 * self._one_phase_step:
            self._last_enough_step = self._total_agent_step
            return True
        else:
            # Get payoff against historical players (Different players have different type of historical players)
            # If min win rate is large enough, then is judge trained enough
            historical = self._get_players(select_fn)
            if len(historical) == 0:
                return False
            win_rates = self._payoff[self, historical]
            if win_rates.min() > self._strong_win_rate:
                self._last_enough_step = self._total_agent_step
                return True
            else:
                return False

    # override
    def get_job(self, p: Optional[np.ndarray] = None) -> dict:
        """
        Overview:
            Additionally get the following job infos:
                - Choose a branch according to prob ``p``, then get an opponent according to the chosen branch.
        Arguments:
            - p (:obj:`np.ndarray`): branch selection probability
        Returns:
            - ret_dict (:obj:`dict`): the job info dict
        """
        parent_dict = super().get_job()
        # select an opponent
        if p is None:
            p = np.random.uniform()
        L = len(self._branch_probs)
        cum_p = [0.] + [sum([j.prob for j in self._branch_probs[:i + 1]]) for i in range(L)]
        idx = [cum_p[i] <= p < cum_p[i + 1] for i in range(L)].index(True)
        branch_name = self._name2branch(self._branch_probs[idx].name)
        opponent = getattr(self, branch_name)()
        my_dict = {'opponent': opponent}
        return deep_merge_dicts(parent_dict, my_dict)

    def _name2branch(self, s: str) -> str:
        """
        Overview:
            Input a branch name and output the corresponding protected method's name, called by ``self.get_job``.
        Arguments:
            - s (:obj:`str`): branch name
        Returns:
            - ret (:obj:`str`): a processed branch name, should be a protected method implemented by ``Player`` \
                or its subclass.
        """
        return '_' + s + '_branch'

    def _get_players(self, select_fn: Callable) -> List[Player]:
        """
        Overview:
            Get a list of players in the league (shared_payoff), selected by ``select_fn`` .
        Arguments:
            - select_fn (:obj:`function`): players in the returned list must satisfy this function
        Returns:
            - players (:obj:`list`): a list of players that satisfies ``select_fn``
        """
        return [player for player in self._payoff.players if select_fn(player)]

    def _get_opponent(self, players: list, p: Optional[list] = None) -> Player:
        """
        Overview:
            Get one opponent player from ``players`` according to probability ``p``.
        Arguments:
            - players (:obj:`list`): a list of players that can select opponent from
            - p (:obj:`list`): the selection probability of each player, should have the same size as ``players``. \
                If you don't need it and set None, it would select uniformly by default.
        Returns:
            - opponent_player (:obj:`list`): a random chosen opponent player according to probability
        """
        idx = np.random.choice(len(players), p=p)
        return players[idx]


class SoloActivePlayer(ActivePlayer):
    """
    Overview:
        Active player class for solo games
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "SoloActivePlayer"

    # override
    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize league player metadata additionally
        Arguments:
            - strong_win_rate (:obj:`float`): if win rates between this player and all the opponents are greater than
                this value, this player can be regarded as strong enough to these opponents, therefore trained enough
            - branch_probs (:obj:`dict`): a dict of probabilities of selecting different opponent branch
        """
        super(SoloActivePlayer, self).__init__(*args, **kwargs)

    # override
    def get_job(self) -> dict:
        """
        Overview:
            Get a dict containing some info about the job to be launched. For example, for solo active player,
            this method can get epsilon value, game mode, scenario, difficulty, etc. For league active player,
            this method can choose an opponent to play against additionally.
        Arguments:
            - exploration_fn (:obj:`function`): the exploration function used for epsilon greedy
        Returns:
            - ret_dict (:obj:`dict`): a dict containing job's epsilon value
        """
        ret_dict = super().get_job()
        return ret_dict
        # epsilon = self.exploration(self.total_agent_step)
        # return deep_merge_dicts(ret_dict, {'epsilon': epsilon})


player_mapping = {}


def register_player(name: str, player: type) -> None:
    """
    Overview:
        Add a new Player class with its name to dict player_mapping, any subclass derived from
        Player must use this function to register in nervex system before instantiate.
    Arguments:
        - name (:obj:`str`): name of the new Player class
        - learner (:obj:`type`): the new Player class, should be subclass of Player
    """
    assert isinstance(name, str)
    assert issubclass(player, Player)
    player_mapping[name] = player


def create_player(cfg: EasyDict, player_type, *args, **kwargs) -> Player:
    """
    Overview:
        Given the key (league_manager_type), create a new league manager instance if in league_mapping's values,
        or raise an KeyError. In other words, a derived league manager must first register then call ``create_league``
        to get the instance object.
    Arguments:
        - cfg (:obj:`EasyDict`): league manager config, necessary keys: [league.import_module]
    Returns:
        - league_manager (:obj:`BaseLeagueManager`): the created new league manager, should be an instance of one of \
            league_mapping's values
    """
    import_module(cfg.import_names)
    # league_type = cfg.league.league_type
    if player_type not in player_mapping.keys():
        raise KeyError("not support player type: {}".format(player_type))
    else:
        return player_mapping[player_type](*args, **kwargs)


register_player('solo_active_player', SoloActivePlayer)
register_player('battle_active_player', BattleActivePlayer)

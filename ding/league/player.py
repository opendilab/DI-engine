from typing import Callable, Optional, List
from collections import namedtuple
import numpy as np
from easydict import EasyDict

from ding.utils import import_module, PLAYER_REGISTRY
from .algorithm import pfsp


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

    def __init__(
            self,
            cfg: EasyDict,
            category: str,
            init_payoff: 'BattleSharedPayoff',  # noqa
            checkpoint_path: str,
            player_id: str,
            total_agent_step: int,
            rating: 'PlayerRating',  # noqa
    ) -> None:
        """
        Overview:
            Initialize base player metadata
        Arguments:
            - cfg (:obj:`EasyDict`): Player config dict.
            - category (:obj:`str`): Player category, depending on the game, \
                e.g. StarCraft has 3 races ['terran', 'protoss', 'zerg'].
            - init_payoff (:obj:`Union[BattleSharedPayoff, SoloSharedPayoff]`): Payoff shared by all players.
            - checkpoint_path (:obj:`str`): The path to load player checkpoint.
            - player_id (:obj:`str`): Player id in string format.
            - total_agent_step (:obj:`int`): For active player, it should be 0; \
                For historical player, it should be parent player's ``_total_agent_step`` when ``snapshot``.
            - rating (:obj:`PlayerRating`): player rating information in total league
        """
        self._cfg = cfg
        self._category = category
        self._payoff = init_payoff
        self._checkpoint_path = checkpoint_path
        assert isinstance(player_id, str)
        self._player_id = player_id
        assert isinstance(total_agent_step, int), (total_agent_step, type(total_agent_step))
        self._total_agent_step = total_agent_step
        self._rating = rating

    @property
    def category(self) -> str:
        return self._category

    @property
    def payoff(self) -> 'BattleSharedPayoff':  # noqa
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

    @property
    def rating(self) -> 'PlayerRating':  # noqa
        return self._rating

    @rating.setter
    def rating(self, _rating: 'PlayerRating') -> None:  # noqa
        self._rating = _rating


@PLAYER_REGISTRY.register('historical_player')
class HistoricalPlayer(Player):
    """
    Overview:
        Historical player which is snapshotted from an active player, and is fixed with the checkpoint.
        Have a unique attribute ``parent_id``.
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
        super().__init__(*args)
        self._parent_id = parent_id

    @property
    def parent_id(self) -> str:
        return self._parent_id


class ActivePlayer(Player):
    """
    Overview:
        Active player can be updated, or snapshotted to a historical player in the league training.
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "ActivePlayer"
    BRANCH = namedtuple("BRANCH", ['name', 'prob'])

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize player metadata, depending on the game
        Note:
            - one_phase_step (:obj:`int`): An active player will be considered trained enough for snapshot \
                after two phase steps.
            - last_enough_step (:obj:`int`): Player's last step number that satisfies ``_is_trained_enough``.
            - strong_win_rate (:obj:`float`): If win rates between this player and all the opponents are greater than
                this value, this player can be regarded as strong enough to these opponents. \
                If also already trained for one phase step, this player can be regarded as trained enough for snapshot.
            - branch_probs (:obj:`namedtuple`): A namedtuple of probabilities of selecting different opponent branch.
        """
        super().__init__(*args)
        self._one_phase_step = int(float(self._cfg.one_phase_step))  # ``one_phase_step`` is like 1e9
        self._last_enough_step = 0
        self._strong_win_rate = self._cfg.strong_win_rate
        assert isinstance(self._cfg.branch_probs, dict)
        self._branch_probs = [self.BRANCH(k, v) for k, v in self._cfg.branch_probs.items()]
        # self._eval_opponent_difficulty = ["WEAK", "MEDIUM", "STRONG"]
        self._eval_opponent_difficulty = ["RULE_BASED"]
        self._eval_opponent_index = 0

    def is_trained_enough(self, select_fn: Optional[Callable] = None) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operations(e.g. snapshot, mutate...)
            according to past step count and overall win rates against opponents.
            If yes, set ``self._last_agent_step`` to ``self._total_agent_step`` and return True; otherwise return False.
        Arguments:
            - select_fn (:obj:`function`): The function to select opponent players.
        Returns:
            - flag (:obj:`bool`): Whether this player is trained enough
        """
        if select_fn is None:
            select_fn = lambda x: isinstance(x, HistoricalPlayer)  # noqa
        step_passed = self._total_agent_step - self._last_enough_step
        if step_passed < self._one_phase_step:
            return False
        elif step_passed >= 2 * self._one_phase_step:
            # ``step_passed`` is 2 times of ``self._one_phase_step``, regarded as trained enough
            self._last_enough_step = self._total_agent_step
            return True
        else:
            # Get payoff against specific opponents (Different players have different type of opponent players)
            # If min win rate is larger than ``self._strong_win_rate``, then is judged trained enough
            selected_players = self._get_players(select_fn)
            if len(selected_players) == 0:  # No such player, therefore no past game
                return False
            win_rates = self._payoff[self, selected_players]
            if win_rates.min() > self._strong_win_rate:
                self._last_enough_step = self._total_agent_step
                return True
            else:
                return False

    def snapshot(self, metric_env: 'LeagueMetricEnv') -> HistoricalPlayer:  # noqa
        """
        Overview:
            Generate a snapshot historical player from the current player, called in league's ``_snapshot``.
        Argument:
            - metric_env (:obj:`LeagueMetricEnv`): player rating environment, one league one env
        Returns:
            - snapshot_player (:obj:`HistoricalPlayer`): new instantiated historical player

        .. note::
            This method only generates a historical player object, but without saving the checkpoint, which should be
            done by league.
        """
        path = self.checkpoint_path.split('.pth')[0] + '_{}'.format(self._total_agent_step) + '.pth'
        return HistoricalPlayer(
            self._cfg,
            self.category,
            self.payoff,
            path,
            self.player_id + '_{}_historical'.format(int(self._total_agent_step)),
            self._total_agent_step,
            metric_env.create_rating(mu=self.rating.mu),
            parent_id=self.player_id
        )

    def mutate(self, info: dict) -> Optional[str]:
        """
        Overview:
            Mutate the current player, called in league's ``_mutate_player``.
        Arguments:
            - info (:obj:`dict`): related information for the mutation
        Returns:
            - mutation_result (:obj:`str`): if the player does the mutation operation then returns the
                corresponding model path, otherwise returns None
        """
        pass

    def get_job(self, eval_flag: bool = False) -> dict:
        """
        Overview:
            Get a dict containing some info about the job to be launched, e.g. the selected opponent.
        Arguments:
            - eval_flag (:obj:`bool`): Whether to select an opponent for evaluator task.
        Returns:
            - ret (:obj:`dict`): The returned dict. Should contain key ['opponent'].
        """
        if eval_flag:
            # eval opponent is a str.
            opponent = self._eval_opponent_difficulty[self._eval_opponent_index]
        else:
            # collect opponent is a Player.
            opponent = self._get_collect_opponent()
        return {
            'opponent': opponent,
        }

    def _get_collect_opponent(self) -> Player:
        """
        Overview:
            Select an opponent according to the player's ``branch_probs``.
        Returns:
            - opponent (:obj:`Player`): Selected opponent.
        """
        p = np.random.uniform()
        L = len(self._branch_probs)
        cum_p = [0.] + [sum([j.prob for j in self._branch_probs[:i + 1]]) for i in range(L)]
        idx = [cum_p[i] <= p < cum_p[i + 1] for i in range(L)].index(True)
        branch_name = '_{}_branch'.format(self._branch_probs[idx].name)
        opponent = getattr(self, branch_name)()
        return opponent

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

    def _get_opponent(self, players: list, p: Optional[np.ndarray] = None) -> Player:
        """
        Overview:
            Get one opponent player from list ``players`` according to probability ``p``.
        Arguments:
            - players (:obj:`list`): a list of players that can select opponent from
            - p (:obj:`np.ndarray`): the selection probability of each player, should have the same size as \
                ``players``. If you don't need it and set None, it would select uniformly by default.
        Returns:
            - opponent_player (:obj:`Player`): a random chosen opponent player according to probability
        """
        idx = np.random.choice(len(players), p=p)
        return players[idx]

    def increment_eval_difficulty(self) -> bool:
        """
        Overview:
            When evaluating, active player will choose a specific builtin opponent difficulty.
            This method is used to increment the difficulty.
            It is usually called after the easier builtin bot is already been beaten by this player.
        Returns:
            - increment_or_not (:obj:`bool`): True means difficulty is incremented; \
                False means difficulty is already the hardest.
        """
        if self._eval_opponent_index < len(self._eval_opponent_difficulty) - 1:
            self._eval_opponent_index += 1
            return True
        else:
            return False

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, path: str) -> None:
        self._checkpoint_path = path


@PLAYER_REGISTRY.register('naive_sp_player')
class NaiveSpPlayer(ActivePlayer):

    def _pfsp_branch(self) -> HistoricalPlayer:
        """
        Overview:
            Select prioritized fictitious self-play opponent, should be a historical player.
        Returns:
            - player (:obj:`HistoricalPlayer`): The selected historical player.
        """
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        win_rates = self._payoff[self, historical]
        # Normal self-play if no historical players
        if win_rates.shape == (0, ):
            return self
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    def _sp_branch(self) -> ActivePlayer:
        """
        Overview:
            Select normal self-play opponent
        """
        return self


def create_player(cfg: EasyDict, player_type: str, *args, **kwargs) -> Player:
    """
    Overview:
        Given the key (player_type), create a new player instance if in player_mapping's values,
        or raise an KeyError. In other words, a derived player must first register then call ``create_player``
        to get the instance object.
    Arguments:
        - cfg (:obj:`EasyDict`): player config, necessary keys: [import_names]
        - player_type (:obj:`str`): the type of player to be created
    Returns:
        - player (:obj:`Player`): the created new player, should be an instance of one of \
            player_mapping's values
    """
    import_module(cfg.get('import_names', []))
    return PLAYER_REGISTRY.build(player_type, *args, **kwargs)

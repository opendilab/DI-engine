from typing import Callable, Optional, List, Union, Any
from collections import namedtuple
import numpy as np
from abc import abstractmethod

from nervex.utils import deep_merge_dicts


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

    def __init__(self, category: str, init_payoff: Union['BattleSharedPayoff', 'SoloSharedPayoff'],  # noqa
                 checkpoint_path: str, player_id: str, total_agent_step: int) -> None:
        """
        Overview: initialize base player metadata
        Arguments:
            - category (:obj:`str`): player category, depending on the game, \
                e.g. StarCraft has 3 races ['terran', 'protoss', 'zerg']
            - init_payoff (:obj:`BattleSharedPayoff` or :obj:`SoloSharedPayoff`): payoff shared by all players
            - checkpoint_path (:obj:`str`): one training phase step
            - player_id (:obj:`str`): player id
            - total_agent_step (:obj:`int`): 0 for active player by default, \
                parent player's ``_total_agent_step`` for historical player
        """
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
    def total_agent_step(self) -> None:
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
        Overview: Initialize ``_parent_id`` additionally
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
    Overview: active player class, active player can be updated
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "ActivePlayer"

    def __init__(self, *args, one_phase_step: int) -> None:
        """
        Overview: initialize player metadata, depending on the game
        Arguments:
            - one_phase_step (:obj:`int`): active player will be considered trained enough after one phase step
        """
        super(ActivePlayer, self).__init__(*args)
        self._one_phase_step = int(float(one_phase_step))  # ``one_phase_step`` is like 1e9
        self._last_enough_step = 0

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
            self.category,
            self.payoff,
            path,
            self.player_id + '_{}'.format(int(self._total_agent_step)),
            self._total_agent_step,
            parent_id=self.player_id
        )

    @abstractmethod
    def mutate(self, info: dict) -> Optional[str]:
        """
        Overview: Mutate the current player
        Arguments:
            - info (:obj:`dict`): related information for the mutation
        Returns:
            - mutation_result (:obj:`str`): if the player does the mutation operation then returns the
                corresponding model path, otherwise returns None
        """
        raise NotImplementedError

    def get_job(self) -> dict:
        """
        Overview:
            Get a dict containing some info about the job to be launched. For example, for solo active player,
            this method can get epsilon value, game mode, scenario, difficulty, etc. For league active player,
            this method can choose an opponent to play against additionally.
        """
        return {}


class BattleActivePlayer(ActivePlayer):
    """
    Overview: active player class for battle games
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "BattleActivePlayer"
    BRANCH = namedtuple("BRANCH", ['name', 'prob'])

    # override
    def __init__(self, *args, strong_win_rate: float, branch_probs: Optional[dict] = None, **kwargs) -> None:
        """
        Overview: Initialize league player metadata additionally
        Arguments:
            - strong_win_rate (:obj:`float`): if win rates between this player and all the opponents are greater than
                this value, this player can be regarded as strong enough to these opponents, therefore trained enough
            - branch_probs (:obj:`dict`): a dict of probabilities of selecting different opponent branch
        """
        super(BattleActivePlayer, self).__init__(*args, **kwargs)
        self._strong_win_rate = strong_win_rate
        assert isinstance(branch_probs, dict)
        self._branch_probs = [self.BRANCH(k, v) for k, v in branch_probs.items()]

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
        ret_dict = super().get_job()
        if p is None:
            p = np.random.uniform()
        L = len(self._branch_probs)
        cum_p = [0.] + [sum([j.prob for j in self._branch_probs[:i + 1]]) for i in range(L)]
        idx = [cum_p[i] <= p < cum_p[i + 1] for i in range(L)].index(True)
        branch_name = self._name2branch(self._branch_probs[idx].name)
        opponent = getattr(self, branch_name)()
        return deep_merge_dicts(ret_dict, {'opponent': opponent})

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
    Overview: active player class for solo games
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    _name = "SoloActivePlayer"

    # override
    def mutate(self, info: dict) -> Optional[str]:
        """
        Overview: Mutate the current player
        Arguments:
            - info (:obj:`dict`): related information for the mutation
        Returns:
            - mutation_result (:obj:`str`): if the player does the mutation operation then returns the
                corresponding model path, otherwise returns None
        """
        pass

    # override
    def get_job(self, exploration_fn: Callable) -> dict:
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
        epsilon = exploration_fn(self.total_agent_step)
        return deep_merge_dicts(ret_dict, {'epsilon': epsilon})

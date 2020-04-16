import numpy as np
from collections import namedtuple
from .algorithm import pfsp


class Player:
    """
    Overview: basic player class, player is the basic member of a league
    Property: race, payoff, checkpoint_path, player_id
    """
    _name = "BasePlayer"

    def __init__(self, race, init_payoff, checkpoint_path, player_id):
        self._race = race
        self._payoff = init_payoff
        self._checkpoint_path = checkpoint_path
        assert isinstance(player_id, str)
        self._player_id = player_id

    @property
    def race(self):
        return self._race

    @property
    def payoff(self):
        """
        Overview: return the current player payoff
        """
        return self._payoff

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def player_id(self):
        return self._player_id


class ActivePlayer(Player):
    """
    Overview: active player class, active player can be updated
    Interface: is_trained_enough, snapshot, mutate, get_match, update_agent_step
    """
    _name = "ActivePlayer"
    BRANCH = namedtuple("BRANCH", ['name', 'prob'])

    def __init__(self, *args, branch_probs, strong_win_rate, one_phase_steps):
        """
        Overview: initialize player metadata
        Arguments:
            - branch_probs (:obj:`list`): a list contains the probabilities of selecting the different opponent branch
            - strong_win_rate (:obj:`float`): if the win rate between this player and the opponent is more than
                this value, this player can be regarded as strong enough to the opponent
             - one_phase_steps (:obj:`int`): one training phase steps
        """
        super(ActivePlayer, self).__init__(*args)
        assert isinstance(branch_probs, dict)
        self._branch_probs = [self.BRANCH(k, v) for k, v in branch_probs.items()]
        self._strong_win_rate = strong_win_rate
        self._one_phase_steps = one_phase_steps
        self._total_agent_steps = 0
        self._last_enough_steps = 0

    def is_trained_enough(self, select_fn):
        """
        Overview: return whether this player is trained enough for further operation
        Arguments:
            - select_fn (:obj:`function`): select players function
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        step_passed = self._total_agent_steps - self._last_enough_steps
        if step_passed < self._one_phase_steps:
            return False
        elif step_passed >= 2 * self._one_phase_steps:
            self._last_enough_steps = self._total_agent_steps
            return True
        else:
            historical = self._get_players(select_fn)
            if len(historical) == 0:
                return False
            win_rates = self._payoff[self, historical]
            if win_rates.min() > self._strong_win_rate:
                self._last_enough_steps = self._total_agent_steps
                return True
            else:
                return False

    def snapshot(self):
        """
        Overview: generate a snapshot of the current player
        Returns:
            - snapshot_player (:obj:`HistoricalPlayer`): snapshot player
        Note:
            this method only generates a player object without saving the checkpoint, which should be completed
            by the interaction between coordinator and learner
        """
        path = self.checkpoint_path.split('.pth')[0] + '_{}'.format(self._total_agent_steps) + '.pth'
        return HistoricalPlayer(
            self.race,
            self.payoff,
            path,
            self.player_id + '_{}'.format(int(self._total_agent_steps)),
            parent_id=self.player_id
        )

    def mutate(self, info):
        """
        Overview: mutate the current player
        Arguments:
            - info (:obj:`dict`): related information for the mutation
        Returns:
            - mutation_result (:obj:`str or None`): if the player does the mutation operation then returns the
                cooresponding model path, otherwise, returns None
        """
        raise NotImplementedError

    def get_match(self, p=None):
        """
        Overview: get an opponent to do a match
        Returns:
            - opponent (:obj:`Player`): match opponent
        """
        if p is None:
            p = np.random.uniform()
        L = len(self._branch_probs)
        cum_p = [0.] + [sum([j.prob for j in self._branch_probs[:i + 1]]) for i in range(L)]
        idx = [cum_p[i] <= p and p < cum_p[i + 1] for i in range(L)].index(True)
        branch_name = self._name2branch(self._branch_probs[idx].name)
        return getattr(self, branch_name)()

    def _name2branch(self, s):
        return '_' + s + '_branch'

    def update_agent_step(self, step):
        """
        Overview: update agent step
        Arguments:
            - step (:obj:`int`): current agent step
        """
        self._total_agent_steps = step

    def _get_players(self, select_fn):
        return [player for player in self._payoff.players if select_fn(player)]

    def _get_opponent(self, players, p=None):
        idx = np.random.choice(len(players), p=p)
        return players[idx]


class HistoricalPlayer(Player):
    """
    Overview: historical player with fixed checkpoint
    Property: parent_id
    """
    _name = "HistoricalPlayer"

    def __init__(self, *args, parent_id):
        super(HistoricalPlayer, self).__init__(*args)
        self._parent_id = parent_id

    @property
    def parent_id(self):
        return self._parent_id


class MainPlayer(ActivePlayer):
    """
    Overview: main player in league training, default branch(0.5 pfsp, 0.35 sp, 0.15 veri)
    """
    _name = "MainPlayer"

    def _pfsp_branch(self):
        """
        Overview: select prioritized fictitious self-play opponent
        Returns:
            - player (:obj:`HistoricalPlayer`): the selected historical player
        """
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        if len(historical) == 0:
            return self._sp_branch()
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    def _sp_branch(self):
        """
        Overview: select normal self-play opponent
        """
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)

        # TODO(nyz) if only one main_player, self-play win_rates are constantly equal to 0.5
        if self._payoff[self, main_opponent] < self._strong_win_rate:
            return main_opponent

        # if the main_opponent is too strong, select a past alternative
        historical = self._get_players(
            lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id
        )
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='variance')
        return self._get_opponent(historical, p)

    def _verification_branch(self):
        """
        Overview: verify no strong main exploiter and no forgetten past main player
        """
        # check exploitation
        main_exploiters = self._get_players(lambda p: isinstance(p, MainExploiter))
        exp_historical = self._get_players(
            lambda p: isinstance(p, HistoricalPlayer) and any([p.parent_id == m.player_id for m in main_exploiters])
        )
        win_rates = self._payoff[self, exp_historical]
        # TODO(nyz) why min win_rates 0.3
        if len(win_rates) and win_rates.min() < 1 - self._strong_win_rate:
            p = pfsp(win_rates, weighting='squared')
            return self._get_opponent(exp_historical, p)

        # check forgetten
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)
        historical = self._get_players(
            lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id
        )
        win_rates = self._payoff[self, historical]
        # TODO(nyz) whether the method `_get_players` should return players with some sequence(such as steps)
        # win_rates, historical = self._remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < self._strong_win_rate:
            p = pfsp(win_rates, weighting='squared')
            return self._get_opponent(historical, p)

        return self._sp_branch()

    def _remove_monotonic_suffix(self, win_rates, players):
        if not len(win_rates):
            return win_rates, players

        for i in range(len(win_rates) - 1, 0, -1):
            if win_rates[i - 1] < win_rates[i]:
                return win_rates[:i + 1], players[:i + 1]

        return np.array([]), []

    # override
    def is_trained_enough(self):
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    # override
    def mutate(self, info):
        """
        Overview: MainPlayer does no mutation
        """
        return None


class MainExploiter(ActivePlayer):
    """
    Overview: main exploiter in league training, default branch(1.0 main_players)
    """
    _name = "MainExploiter"

    def __init__(self, *args, min_valid_win_rate=0.2, **kwargs):
        super(MainExploiter, self).__init__(*args, **kwargs)
        self._min_valid_win_rate = min_valid_win_rate

    def _main_players_branch(self):
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)

        # if this main_opponent can produce valid training signals
        if self._payoff[self, main_opponent] >= self._min_valid_win_rate:
            return main_opponent

        # otherwise, curriculum learning
        historical = self._get_players(
            lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id
        )
        if len(historical) == 0:
            return main_opponent
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='variance')
        return self._get_opponent(historical)

    # override
    def is_trained_enough(self):
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, MainPlayer))

    # override
    def mutate(self, info):
        """
        Overview: main exploiter is sure to mutates(reset) to the supervised learning player
        """
        return info['sl_checkpoint_path']


class LeagueExploiter(ActivePlayer):
    """
    Overview: league exploiter in league training, default branch(1.0 pfsp)
    """
    _name = "LeagueExploiter"

    def _pfsp_branch(self):
        """
        Overview: select prioritized fictitious self-play opponent
        Returns:
            - player (:obj:`HistoricalPlayer`): the selected historical player
        Note:
            This branch is the same as the psfp branch in MainPlayer
        """
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        if len(historical) == 0:
            return self._sp_branch()
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    # override
    def is_trained_enough(self):
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    # override
    def mutate(self, info):
        """
        Overview: league exploiter can mutate to the supervised learning player with 0.25 prob
        """
        p = np.random.uniform()
        if p < 0.25:
            return info['sl_checkpoint_path']

        return None

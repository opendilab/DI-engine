from typing import Optional, Union
import numpy as np

from ding.utils import PLAYER_REGISTRY
from .player import ActivePlayer, HistoricalPlayer
from .algorithm import pfsp


@PLAYER_REGISTRY.register('main_player')
class MainPlayer(ActivePlayer):
    """
    Overview:
        Main player in league training.
        Default branch (0.5 pfsp, 0.35 sp, 0.15 veri).
        Default snapshot every 2e9 steps.
        Default mutate prob = 0 (never mutate).
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_iteration
    """
    _name = "MainPlayer"

    def _pfsp_branch(self) -> HistoricalPlayer:
        """
        Overview:
            Select prioritized fictitious self-play opponent, should be a historical player.
        Returns:
            - player (:obj:`HistoricalPlayer`): the selected historical player
        """
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    def _sp_branch(self):
        """
        Overview:
            Select normal self-play opponent
        """
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)

        # TODO(nyz) if only one main_player, self-play win_rates are constantly equal to 0.5
        # main_opponent is not too strong
        if self._payoff[self, main_opponent] > 1 - self._strong_win_rate:
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
        Overview:
            Verify no strong historical main exploiter and no forgotten historical past main player
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

        # check forgotten
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)  # only one main player
        main_historical = self._get_players(
            lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id
        )
        win_rates = self._payoff[self, main_historical]
        # TODO(nyz) whether the method `_get_players` should return players with some sequence(such as step)
        # win_rates, historical = self._remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < self._strong_win_rate:
            p = pfsp(win_rates, weighting='squared')
            return self._get_opponent(main_historical, p)

        # no forgotten main players or strong main exploiters, use self-play instead
        return self._sp_branch()

    # def _remove_monotonic_suffix(self, win_rates, players):
    #     if not len(win_rates):
    #         return win_rates, players
    #     for i in range(len(win_rates) - 1, 0, -1):
    #         if win_rates[i - 1] < win_rates[i]:
    #             return win_rates[:i + 1], players[:i + 1]
    #     return np.array([]), []

    # override
    def is_trained_enough(self) -> bool:
        # ``_pfsp_branch`` and ``_verification_branch`` are played against historcial player
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    # override
    def mutate(self, info: dict) -> None:
        """
        Overview:
            MainPlayer does not mutate
        """
        pass


@PLAYER_REGISTRY.register('main_exploiter')
class MainExploiter(ActivePlayer):
    """
    Overview:
        Main exploiter in league training. Can identify weaknesses of main agents, and consequently make them
        more robust.
        Default branch (1.0 main_players).
        Default snapshot when defeating all 3 main players in the league in more than 70% of games,
        or timeout of 4e9 steps.
        Default mutate prob = 1 (must mutate).
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_iteration
    """
    _name = "MainExploiter"

    def __init__(self, *args, **kwargs):
        """
        Overview:
            Initialize ``min_valid_win_rate`` additionally
        Note:
            - min_valid_win_rate (:obj:`float`): only when win rate against the main player is greater than this, \
                can the main player be regarded as able to produce valid training signals to be selected
        """
        super(MainExploiter, self).__init__(*args, **kwargs)
        self._min_valid_win_rate = self._cfg.min_valid_win_rate

    def _main_players_branch(self):
        """
        Overview:
            Select main player or historical player snapshot from main player as opponent
        Returns:
            - player (:obj:`Player`): the selected main player (active/historical)
        """
        # get the main player (only one)
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)
        # if this main_opponent can produce valid training signals
        if self._payoff[self, main_opponent] >= self._min_valid_win_rate:
            return main_opponent
        # otherwise, curriculum learning, select a historical version
        historical = self._get_players(
            lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id
        )
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='variance')
        return self._get_opponent(historical, p)

    # override
    def is_trained_enough(self):
        # would play against main player, or historical main player (if main player is too strong)
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, MainPlayer))

    # override
    def mutate(self, info: dict) -> str:
        """
        Overview:
            Main exploiter is sure to mutate(reset) to the supervised learning player
        Returns:
            - mutate_ckpt_path (:obj:`str`): mutation target checkpoint path
        """
        return info['reset_checkpoint_path']


@PLAYER_REGISTRY.register('league_exploiter')
class LeagueExploiter(ActivePlayer):
    """
    Overview:
        League exploiter in league training. Can identify global blind spots in the league (strategies that no player
        in the league can beat, but that are not necessarily robust themselves).
        Default branch (1.0 pfsp).
        Default snapshot when defeating all players in the league in more than 70% of games, or timeout of 2e9 steps.
        Default mutate prob = 0.25.
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_iteration
    """
    _name = "LeagueExploiter"

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize ``mutate_prob`` additionally
        Note:
            - mutate_prob (:obj:`float`): the mutation probability of league exploiter. should be in [0, 1]
        """
        super(LeagueExploiter, self).__init__(*args, **kwargs)
        assert 0 <= self._cfg.mutate_prob <= 1
        self.mutate_prob = self._cfg.mutate_prob

    def _pfsp_branch(self) -> HistoricalPlayer:
        """
        Overview:
            Select prioritized fictitious self-play opponent
        Returns:
            - player (:obj:`HistoricalPlayer`): the selected historical player
        Note:
            This branch is the same as the psfp branch in MainPlayer
        """
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    # override
    def is_trained_enough(self) -> bool:
        # will only player against historical player
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    # override
    def mutate(self, info) -> Union[str, None]:
        """
        Overview:
            League exploiter can mutate to the supervised learning player with 0.25 prob
        Returns:
            - ckpt_path (:obj:`Union[str, None]`): with ``mutate_prob`` prob returns the pretrained model's ckpt path, \
                with left 1 - ``mutate_prob`` prob returns None, which means no mutation
        """
        p = np.random.uniform()
        if p < self.mutate_prob:
            return info['reset_checkpoint_path']
        return None

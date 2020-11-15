from typing import Optional
import numpy as np

from .player import BattleActivePlayer, HistoricalPlayer
from .algorithm import pfsp


class MainPlayer(BattleActivePlayer):
    """
    Overview:
        Main player in league training, default branch(0.5 pfsp, 0.35 sp, 0.15 veri)
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_step
    """
    _name = "MainPlayer"

    def _pfsp_branch(self) -> HistoricalPlayer:
        """
        Overview: Select prioritized fictitious self-play opponent
        Returns:
            - player (:obj:`HistoricalPlayer`): the selected historical player
        """
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    def _sp_branch(self):
        """
        Overview: Select normal self-play opponent
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
        Overview: verify no strong main exploiter and no forgotten past main player
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
        main_opponent = self._get_opponent(main_players)
        historical = self._get_players(
            lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id
        )
        win_rates = self._payoff[self, historical]
        # TODO(nyz) whether the method `_get_players` should return players with some sequence(such as step)
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
    def is_trained_enough(self) -> bool:
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    # override
    def mutate(self, info: dict) -> None:
        """
        Overview: MainPlayer does no mutation
        """
        return None


class MainExploiter(BattleActivePlayer):
    """
    Overview:
        Main exploiter in league training, default branch(1.0 main_players)
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_step
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
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='variance')
        return self._get_opponent(historical)

    # override
    def is_trained_enough(self):
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, MainPlayer))

    # override
    def mutate(self, info: dict) -> str:
        """
        Overview: main exploiter is sure to mutates(reset) to the supervised learning player
        Returns:
            - ckpt_path (:obj:`str`): the pretrained model's ckpt path
        """
        return info['pretrain_checkpoint_path']


class LeagueExploiter(BattleActivePlayer):
    """
    Overview:
        League exploiter in league training, default branch(1.0 pfsp)
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_step
    """
    _name = "LeagueExploiter"

    def __init__(self, *args, mutate_prob: float = 0.25, **kwargs) -> None:
        """
        Overview: Initialize ``mutate_prob`` additionally
        Returns:
            - mutate_prob (:obj:`str`): the mutation probability of league exploiter. should be in [0, 1]
        """
        super(LeagueExploiter, self).__init__(*args, **kwargs)
        assert 0 <= mutate_prob <= 1
        self.mutate_prob = mutate_prob

    def _pfsp_branch(self) -> HistoricalPlayer:
        """
        Overview: select prioritized fictitious self-play opponent
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
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    # override
    def mutate(self, info) -> Optional[str]:
        """
        Overview: League exploiter can mutate to the supervised learning player with 0.25 prob
        Returns:
            - ckpt_path (:obj:`str`): with 0.25 prob returns the pretrained model's ckpt path, \
                with left 0.75 prob returns None, which means no mutation
        """
        p = np.random.uniform()
        if p < self.mutate_prob:
            return info['pretrain_checkpoint_path']
        return None

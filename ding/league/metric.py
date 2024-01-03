from typing import Tuple, Union, List
import math
import numpy as np
from trueskill import TrueSkill, Rating, rate_1vs1


class EloCalculator(object):
    """
    Overview:
        A class that calculates Elo ratings for players based on game results.

    Attributes:
        - score (:obj:`dict`): A dictionary that maps game results to scores.

    Interfaces:
        ``__init__``, ``get_new_rating``, ``get_new_rating_array``.
    """

    score = {
        1: 1.0,  # win
        0: 0.5,  # draw
        -1: 0.0,  # lose
    }

    @classmethod
    def get_new_rating(cls,
                       rating_a: int,
                       rating_b: int,
                       result: int,
                       k_factor: int = 32,
                       beta: int = 200) -> Tuple[int, int]:
        """
        Overview:
            Calculates the new ratings for two players based on their current ratings and game result.

        Arguments:
            - rating_a (:obj:`int`): The current rating of player A.
            - rating_b (:obj:`int`): The current rating of player B.
            - result (:obj:`int`): The result of the game: 1 for player A win, 0 for draw, -1 for player B win.
            - k_factor (:obj:`int`): The K-factor used in the Elo rating system. Defaults to 32.
            - beta (:obj:`int`): The beta value used in the Elo rating system. Defaults to 200.

        Returns:
            -ret (:obj:`Tuple[int, int]`): The new ratings for player A and player B, respectively.
        """
        assert result in [1, 0, -1]
        expect_a = 1. / (1. + math.pow(10, (rating_b - rating_a) / (2. * beta)))
        expect_b = 1. / (1. + math.pow(10, (rating_a - rating_b) / (2. * beta)))
        new_rating_a = rating_a + k_factor * (EloCalculator.score[result] - expect_a)
        new_rating_b = rating_b + k_factor * (1 - EloCalculator.score[result] - expect_b)
        return round(new_rating_a), round(new_rating_b)

    @classmethod
    def get_new_rating_array(
            cls,
            rating: np.ndarray,
            result: np.ndarray,
            game_count: np.ndarray,
            k_factor: int = 32,
            beta: int = 200
    ) -> np.ndarray:
        """
        Overview:
            Calculates the new ratings for multiple players based on their current ratings, game results, \
            and game counts.

        Arguments:
            - rating (obj:`np.ndarray`): An array of current ratings for each player.
            - result (obj:`np.ndarray`): An array of game results, where 1 represents a win, 0 represents a draw, \
                and -1 represents a loss.
            - game_count (obj:`np.ndarray`): An array of game counts for each player.
            - k_factor (obj:`int`): The K-factor used in the Elo rating system. Defaults to 32.
            - beta (obj:`int`): The beta value used in the Elo rating system. Defaults to 200.

        Returns:
            -ret(obj:`np.ndarray`): An array of new ratings for each player.

        Shapes:
            - rating (obj:`np.ndarray`): :math:`(N, )`, N is the number of player
            - result (obj:`np.ndarray`): :math:`(N, N)`
            - game_count (obj:`np.ndarray`): :math:`(N, N)`
        """
        rating_diff = np.expand_dims(rating, 0) - np.expand_dims(rating, 1)
        expect = 1. / (1. + np.power(10, rating_diff / (2. * beta))) * game_count
        delta = ((result + 1.) / 2 - expect) * (game_count > 0)
        delta = delta.sum(axis=1)
        return np.round(rating + k_factor * delta).astype(np.int64)


class PlayerRating(Rating):
    """
    Overview:
        Represents the rating of a player.

    Interfaces:
        ``__init__``, ``__repr__``.
    """

    def __init__(self, mu: float = None, sigma: float = None, elo_init: int = None) -> None:
        super(PlayerRating, self).__init__(mu, sigma)
        self.elo = elo_init

    def __repr__(self) -> str:
        c = type(self)
        args = ('.'.join([c.__module__, c.__name__]), self.mu, self.sigma, self.exposure, self.elo)
        return '%s(mu=%.3f, sigma=%.3f, exposure=%.3f, elo=%d)' % args


class LeagueMetricEnv(TrueSkill):
    """
    Overview:
        A class that represents a TrueSkill rating system for game players. Inherits from the TrueSkill class. \
        For more details, please refer to https://trueskill.org/.

    Interfaces:
        ``__init__``, ``create_rating``, ``rate_1vs1``, ``rate_1vsC``.
    """

    def __init__(self, *args, elo_init: int = 1200, **kwargs) -> None:
        super(LeagueMetricEnv, self).__init__(*args, **kwargs)
        self.elo_init = elo_init

    def create_rating(self, mu: float = None, sigma: float = None, elo_init: int = None) -> PlayerRating:
        """
        Overview:
            Creates a new player rating object with the specified mean, standard deviation, and Elo rating.

        Arguments:
            - mu (:obj:`float`): The mean value of the player's skill rating. If not provided, the default \
                TrueSkill mean is used.
            - sigma (:obj:`float`): The standard deviation of the player's skill rating. If not provided, \
                the default TrueSkill sigma is used.
            - elo_init (:obj:int`): The initial Elo rating value for the player. If not provided, the default \
                elo_init value of the LeagueMetricEnv class is used.

        Returns:
            - PlayerRating: A player rating object with the specified mean, standard deviation, and Elo rating.
        """
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if elo_init is None:
            elo_init = self.elo_init
        return PlayerRating(mu, sigma, elo_init)

    @staticmethod
    def _rate_1vs1(t1, t2, **kwargs):
        t1_elo, t2_elo = t1.elo, t2.elo
        t1, t2 = rate_1vs1(t1, t2, **kwargs)
        if 'drawn' in kwargs:
            result = 0
        else:
            result = 1
        t1_elo, t2_elo = EloCalculator.get_new_rating(t1_elo, t2_elo, result)
        t1 = PlayerRating(t1.mu, t1.sigma, t1_elo)
        t2 = PlayerRating(t2.mu, t2.sigma, t2_elo)
        return t1, t2

    def rate_1vs1(self, team1: PlayerRating, team2: PlayerRating, result: List[str] = None, **kwargs) \
            -> Tuple[PlayerRating, PlayerRating]:
        """
        Overview:
            Rates two teams of players against each other in a 1 vs 1 match and returns the updated ratings \
                for both teams.

        Arguments:
            - team1 (:obj:`PlayerRating`): The rating object representing the first team of players.
            - team2 (:obj:`PlayerRating`): The rating object representing the second team of players.
            - result (:obj:`List[str]`): The result of the match. Can be 'wins', 'draws', or 'losses'. If \
                not provided, the default behavior is to rate the match as a win for team1.

        Returns:
            - ret (:obj:`Tuple[PlayerRating, PlayerRating]`): A tuple containing the updated ratings for team1 \
                and team2.
        """
        if result is None:
            return self._rate_1vs1(team1, team2, **kwargs)
        else:
            for r in result:
                if r == 'wins':
                    team1, team2 = self._rate_1vs1(team1, team2)
                elif r == 'draws':
                    team1, team2 = self._rate_1vs1(team1, team2, drawn=True)
                elif r == 'losses':
                    team2, team1 = self._rate_1vs1(team2, team1)
                else:
                    raise RuntimeError("invalid result: {}".format(r))
        return team1, team2

    def rate_1vsC(self, team1: PlayerRating, team2: PlayerRating, result: List[str]) -> PlayerRating:
        """
        Overview:
            Rates a team of players against a single player in a 1 vs C match and returns the updated rating \
            for the team.

        Arguments:
            - team1 (:obj:`PlayerRating`): The rating object representing the team of players.
            - team2 (:obj:`PlayerRating`): The rating object representing the single player.
            - result (:obj:`List[str]`): The result of the match. Can be 'wins', 'draws', or 'losses'.

        Returns:
            - PlayerRating: The updated rating for the team of players.
        """
        for r in result:
            if r == 'wins':
                team1, _ = self._rate_1vs1(team1, team2)
            elif r == 'draws':
                team1, _ = self._rate_1vs1(team1, team2, drawn=True)
            elif r == 'losses':
                _, team1 = self._rate_1vs1(team2, team1)
            else:
                raise RuntimeError("invalid result: {}".format(r))
        return team1


get_elo = EloCalculator.get_new_rating
get_elo_array = EloCalculator.get_new_rating_array

import pytest
import numpy as np
from trueskill import TrueSkill, rate_1vs1
from ding.league import get_elo, get_elo_array


@pytest.mark.unittest
def test_elo_calculator():
    game_count = np.array([[0, 1, 2], [1, 0, 0], [2, 0, 0]])
    rating = np.array([1613, 1573, 1601])
    result = np.array([[0, -1, -1 + 1], [1, 0, 0], [1 + (-1), 0, 0]])
    new_rating0, new_rating1 = get_elo(rating[0], rating[1], result[0][1])
    assert new_rating0 == 1595
    assert new_rating1 == 1591

    old_rating = np.copy(rating)
    new_rating = get_elo_array(rating, result, game_count)
    assert (rating == old_rating).all()  # no inplace modification
    assert new_rating.dtype == np.int64
    assert new_rating[0] == 1578
    assert new_rating[1] == 1591
    assert new_rating[2] == 1586


@pytest.mark.unittest
def test_true_skill():
    sigma = 25 / 3
    env = TrueSkill(mu=0, sigma=sigma, beta=sigma / 2, tau=0.0, draw_probability=0.02)
    r1 = env.create_rating()
    r2 = env.create_rating()
    assert r1.mu == 0
    assert r2.mu == 0
    assert r2.sigma == sigma
    assert r2.sigma == sigma
    # r1 draw r2
    r1, r2 = rate_1vs1(r1, r2, drawn=True)
    assert r1.mu == r2.mu
    # r1 win r2
    new_r1, new_r2 = rate_1vs1(r1, r2)
    assert new_r1.mu > r1.mu
    assert new_r2.mu < r2.mu
    assert new_r1.mu + new_r2.mu == 0
    assert pytest.approx(new_r1.mu, 3.230)
    assert pytest.approx(new_r2.mu, -3.230)

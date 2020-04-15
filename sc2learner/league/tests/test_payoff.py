import pytest
import numpy as np
from sc2learner.league.payoff import Payoff
from sc2learner.league.player import Player

@pytest.fixture(scope='function')
def setup_payoff():
    return Payoff(home_id='test_payoff_0', decay=0.99)


global fake_player_count
fake_player_count = 0


def get_fake_player():
    global fake_player_count
    player = Player(
        race='zerg',
        init_payoff=None,
        checkpoint_path='fake_ckpt_{}.pth'.format(fake_player_count),
        player_id='fake_player_{}'.format(fake_player_count)
    )
    fake_player_count += 1
    return player


def random_match_result():
    p = np.random.uniform()
    if p < 1./3:
        return "wins"
    elif p < 2./3:
        return "draws"
    else:
        return "losses"


@pytest.mark.unittest
class TestPayoff:
    def test_add_player(self, setup_payoff):
        assert len(setup_payoff.players) == 0
        N = 10
        keys_set = set()
        player_list = []
        for n in range(N):
            p = get_fake_player()
            player_list.append(p)
            setup_payoff.add_player(p)
            keys_set.add(p.player_id)
            assert len(setup_payoff.players) == n + 1
        assert len(setup_payoff.players) == N
        assert keys_set == set(setup_payoff._data.keys())
        for key in keys_set:
            for k in Payoff.data_keys:
                assert setup_payoff._data[key][k] == 0

        # test shallow copy(only reference)
        for p_ori, p_copy in zip(player_list, setup_payoff.players):
            assert id(p_ori) == id(p_copy)
            p_ori._payoff = np.random.uniform()
            assert p_ori.payoff == p_copy.payoff

    def test_update(self, setup_payoff):
        assert len(setup_payoff.players) == 0
        N = 10
        games_per_player = 8
        player_list = [get_fake_player() for _ in range(N)]
        for p in player_list:
            setup_payoff.add_player(p)

        for n in range(N):
            away = setup_payoff.players[n].player_id
            for i in range(games_per_player):
                match_result = random_match_result()
                match_info = {
                    'home': setup_payoff._home_id,
                    'away': away,
                    'result': match_result,
                }
                old = setup_payoff._data[away][match_result]
                result = setup_payoff.update(match_info)
                assert result
                assert old*setup_payoff._decay + 1 == setup_payoff._data[away][match_result]

        # invalid update test
        match_info = None
        assert not setup_payoff.update(match_info)

    def test_getitem(self, setup_payoff):
        assert len(setup_payoff.players) == 0
        N = 10
        match_num = 314
        player_list = [get_fake_player() for _ in range(N)]
        for p in player_list:
            setup_payoff.add_player(p)

        for i in range(match_num):
            away = np.random.choice(setup_payoff.players).player_id
            match_info = {
                'home': setup_payoff._home_id,
                'away': away,
                'result': random_match_result(),
            }
            setup_payoff.update(match_info)
        # single player
        idx = np.random.randint(0, len(setup_payoff.players))
        player = setup_payoff.players[idx]
        win_rates = setup_payoff[player]
        assert isinstance(win_rates, np.ndarray)
        handle = setup_payoff._data[player.player_id]
        if handle['games'] > 1e-6:
            assert win_rates[0] == pytest.approx((handle['wins'] + 0.5*handle['draws']) / handle['games'])
        else:
            assert win_rates[0] == pytest.approx(0.5)

        # players list
        idxes = np.random.choice(range(len(setup_payoff.players)), 5)
        players = [setup_payoff.players[idx] for idx in idxes]
        win_rates = setup_payoff[players]
        for win_rate, player in zip(win_rates, players):
            handle = setup_payoff._data[player.player_id]
            if handle['games'] > 1e-6:
                assert win_rate == pytest.approx((handle['wins'] + 0.5*handle['draws']) / handle['games'])
            else:
                assert win_rate == pytest.approx(0.5)

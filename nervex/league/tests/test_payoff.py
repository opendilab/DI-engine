import pytest
import numpy as np
from nervex.league.payoff import Payoff
from nervex.league.shared_payoff import PayoffDict, SharedPayoff
from nervex.league.player import Player


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

    def test_update(self, setup_payoff, random_match_result):
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
                    'home_id': setup_payoff._home_id,
                    'away_id': away,
                    'result': match_result,
                }
                old = setup_payoff._data[away][match_result]
                result = setup_payoff.update(match_info)
                assert result
                assert old * setup_payoff._decay + 1 == setup_payoff._data[away][match_result]

        # invalid update test
        match_info = None
        assert not setup_payoff.update(match_info)

    def test_getitem(self, setup_payoff, random_match_result):
        assert len(setup_payoff.players) == 0
        N = 10
        match_num = 314
        player_list = [get_fake_player() for _ in range(N)]
        for p in player_list:
            setup_payoff.add_player(p)

        for i in range(match_num):
            away = np.random.choice(setup_payoff.players).player_id
            match_info = {
                'home_id': setup_payoff._home_id,
                'away_id': away,
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
            assert win_rates[0] == pytest.approx((handle['wins'] + 0.5 * handle['draws']) / handle['games'])
        else:
            assert win_rates[0] == pytest.approx(0.5)

        # players list
        idxes = np.random.choice(range(len(setup_payoff.players)), 5)
        players = [setup_payoff.players[idx] for idx in idxes]
        win_rates = setup_payoff[players]
        for win_rate, player in zip(win_rates, players):
            handle = setup_payoff._data[player.player_id]
            if handle['games'] > 1e-6:
                assert win_rate == pytest.approx((handle['wins'] + 0.5 * handle['draws']) / handle['games'])
            else:
                assert win_rate == pytest.approx(0.5)


@pytest.mark.unittest
class TestPayoffDict:
    def test_init(self):
        data = PayoffDict()
        data['test_player_0-test_player_1'] *= 1
        assert data['test_player_0-test_player_1']['wins'] == 0
        assert data['test_player_0-test_player_1']['draws'] == 0
        assert data['test_player_0-test_player_1']['losses'] == 0
        assert data['test_player_0-test_player_1']['games'] == 0
        with pytest.raises(KeyError):
            tmp = data['test_player_0-test_player_1']['xxx']


@pytest.fixture(scope='function')
def setup_shared_payoff():
    return SharedPayoff(decay=0.99)


global sp_player_count
sp_player_count = 0


def get_shared_payoff_player(payoff):
    global sp_player_count
    player = Player(
        race='zerg',
        init_payoff=payoff,
        checkpoint_path='sp_ckpt_{}.pth'.format(sp_player_count),
        player_id='sp_player_{}'.format(sp_player_count)
    )
    sp_player_count += 1
    return player


@pytest.mark.unittest
class TestSharedPayoff:
    def test_update(self, setup_shared_payoff, random_match_result):
        N = 10
        games_per_player = 4
        player_list = [get_shared_payoff_player(setup_shared_payoff) for _ in range(N)]
        for p in player_list:
            setup_shared_payoff.add_player(p)

        for home in player_list:
            for away in player_list:
                for i in range(games_per_player):
                    match_result = random_match_result()
                    match_info = {
                        'home_id': home.player_id,
                        'away_id': away.player_id,
                        'result': match_result,
                    }
                    key = setup_shared_payoff.get_key(home.player_id, away.player_id)
                    if key in setup_shared_payoff._data.keys():
                        old = setup_shared_payoff._data[key][match_result]
                    else:
                        old = 0
                    result = setup_shared_payoff.update(match_info)
                    assert result
                    assert old * setup_shared_payoff._decay + 1 == setup_shared_payoff._data[key][match_result]

        # test shared payoff
        for p in player_list:
            assert id(p.payoff) == id(setup_shared_payoff)

    def test_getitem(self, setup_shared_payoff, random_match_result):
        N = 10
        games_per_player = 4
        player_list = [get_shared_payoff_player(setup_shared_payoff) for _ in range(N)]
        for p in player_list:
            setup_shared_payoff.add_player(p)

        # test key not in setup_shared_payoff._data
        home = player_list[0]
        away = player_list[0]
        key = setup_shared_payoff.get_key(home.player_id, away.player_id)
        assert key not in setup_shared_payoff._data.keys()
        win_rate = setup_shared_payoff[home, away]
        assert key in setup_shared_payoff._data.keys()
        assert len(win_rate.shape) == 1
        assert win_rate[0] == pytest.approx(0.5)

        # test playes list
        for i in range(314):
            home = np.random.choice(setup_shared_payoff.players)
            away = np.random.choice(setup_shared_payoff.players)
            match_info = {'home_id': home.player_id, 'away_id': away.player_id, 'result': random_match_result()}
            result = setup_shared_payoff.update(match_info)
            assert result
        for i in range(314):
            home_num = np.random.randint(1, N + 1)
            home = np.random.choice(setup_shared_payoff.players, home_num).tolist()
            away_num = np.random.randint(1, N + 1)
            away = np.random.choice(setup_shared_payoff.players, away_num).tolist()
            win_rates = setup_shared_payoff[home, away]
            assert isinstance(win_rates, np.ndarray)
            if home_num == 1 or away_num == 1:
                assert len(win_rates.shape) == 1
            else:
                assert len(win_rates.shape) == 2
                assert win_rates.shape == (home_num, away_num)
            assert win_rates.max() <= 1.
            assert win_rates.min() >= 0.

        # test shared payoff
        for p in player_list:
            assert id(p.payoff) == id(setup_shared_payoff)

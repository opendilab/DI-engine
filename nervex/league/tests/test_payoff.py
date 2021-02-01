import numpy as np
from easydict import EasyDict
import pytest
from copy import deepcopy
from collections import deque
from functools import partial
import os
import yaml

from nervex.league.payoff import Payoff
from nervex.league.player import Player, SoloActivePlayer
from nervex.league.shared_payoff import BattleRecordDict, PayoffDict, \
    BattleSharedPayoff, SoloSharedPayoff, create_payoff


@pytest.fixture(scope='function')
def setup_payoff():
    return Payoff(home_id='test_payoff_0', decay=0.99)


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), 'league_test_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


global fake_player_count
fake_player_count = 0


def get_fake_player():
    global fake_player_count
    player = Player(
        cfg=EasyDict(),
        category='zerg',
        init_payoff=None,
        checkpoint_path='fake_ckpt_{}.pth'.format(fake_player_count),
        player_id='fake_player_{}'.format(fake_player_count),
        total_agent_step=0
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

    def test_update(self, setup_payoff, random_job_result):
        assert len(setup_payoff.players) == 0
        N = 10
        games_per_player = 8
        player_list = [get_fake_player() for _ in range(N)]
        for p in player_list:
            setup_payoff.add_player(p)

        for n in range(N):
            away = setup_payoff.players[n].player_id
            for i in range(games_per_player):
                job_result = random_job_result()
                job_info = {
                    'home_id': setup_payoff._home_id,
                    'away_id': away,
                    'result': job_result,
                }
                old = setup_payoff._data[away][job_result]
                result = setup_payoff.update(job_info)
                assert result
                assert old * setup_payoff._decay + 1 == setup_payoff._data[away][job_result]

        # invalid update test
        job_info = None
        assert not setup_payoff.update(job_info)

    def test_getitem(self, setup_payoff, random_job_result):
        assert len(setup_payoff.players) == 0
        N = 10
        job_num = 314
        player_list = [get_fake_player() for _ in range(N)]
        for p in player_list:
            setup_payoff.add_player(p)

        for i in range(job_num):
            away = np.random.choice(setup_payoff.players).player_id
            job_info = {
                'home_id': setup_payoff._home_id,
                'away_id': away,
                'result': random_job_result(),
            }
            setup_payoff.update(job_info)
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
        data1 = PayoffDict(BattleRecordDict)
        data1['test_player_0-test_player_1'] *= 1
        assert data1['test_player_0-test_player_1']['wins'] == 0
        assert data1['test_player_0-test_player_1']['draws'] == 0
        assert data1['test_player_0-test_player_1']['losses'] == 0
        assert data1['test_player_0-test_player_1']['games'] == 0
        with pytest.raises(KeyError):
            tmp = data1['test_player_0-test_player_1']['xxx']
        # data2 = PayoffDict(SoloRecordQueue, EasyDict({'buffer_size': 1}))
        data2 = PayoffDict(partial(deque, maxlen=1))
        data2['test_player_0'].append(3)
        assert len(data2['test_player_0']) == 1
        data2['test_player_0'].append(5)
        assert len(data2['test_player_0']) == 1
        assert data2['test_player_0'][0] == 5


@pytest.fixture(scope='function')
def setup_battle_shared_payoff():
    cfg = EasyDict({'type': 'battle', 'decay': 0.99})
    return create_payoff(cfg)


@pytest.fixture(scope='function')
def setup_solo_shared_payoff():
    cfg = EasyDict({'type': 'solo', 'buffer_size': 3})
    return create_payoff(cfg)


global sp_player_count
sp_player_count = 0


def get_shared_payoff_player(payoff):
    global sp_player_count
    player = Player(
        cfg=EasyDict(),
        category='zerg',
        init_payoff=payoff,
        checkpoint_path='sp_ckpt_{}.pth'.format(sp_player_count),
        player_id='sp_player_{}'.format(sp_player_count),
        total_agent_step=0
    )
    sp_player_count += 1
    return player


def _win_loss_reverse(result_: str, reverse_: bool) -> str:
    if result_ == 'draws' or not reverse_:
        return result_
    reverse_dict = {'wins': 'losses', 'losses': 'wins'}
    return reverse_dict[result_]


@pytest.mark.unittest
class TestBattleSharedPayoff:

    def test_update(self, setup_battle_shared_payoff, random_job_result, get_job_result_categories):
        N = 10
        games_per_player = 4
        player_list = [get_shared_payoff_player(setup_battle_shared_payoff) for _ in range(N)]
        for p in player_list:
            setup_battle_shared_payoff.add_player(p)

        # test update exception
        job_info = {
            'player_id': [player_list[0].player_id, player_list[1].player_id],
            'episode_num': 1,
            'env_num': 1,
            'result': [["error"]]
        }
        assert not setup_battle_shared_payoff.update(job_info)

        for home in player_list:
            for away in player_list:
                for i in range(games_per_player):
                    episode_num = 2
                    env_num = 4
                    job_result = [[random_job_result() for _ in range(env_num)] for _ in range(episode_num)]
                    job_info = {
                        'player_id': [home.player_id, away.player_id],
                        'episode_num': episode_num,
                        'env_num': env_num,
                        'result': job_result
                    }
                    key, reverse = setup_battle_shared_payoff.get_key(home.player_id, away.player_id)
                    old = deepcopy(setup_battle_shared_payoff._data[key])
                    assert setup_battle_shared_payoff.update(job_info)

                    decay = setup_battle_shared_payoff._decay
                    for j in job_result:
                        for i in j:
                            for k in get_job_result_categories:
                                old[k] *= decay
                            result = _win_loss_reverse(i, reverse)
                            old[result] += 1

                    for t in get_job_result_categories:
                        assert old[t] == setup_battle_shared_payoff._data[key][t], t

        # test shared payoff
        for p in player_list:
            assert id(p.payoff) == id(setup_battle_shared_payoff)

    def test_getitem(self, setup_battle_shared_payoff, random_job_result):
        N = 10
        games_per_player = 4
        player_list = [get_shared_payoff_player(setup_battle_shared_payoff) for _ in range(N)]
        for p in player_list:
            setup_battle_shared_payoff.add_player(p)

        # test key not in setup_battle_shared_payoff._data
        home = player_list[0]
        away = player_list[0]
        key, reverse = setup_battle_shared_payoff.get_key(home.player_id, away.player_id)
        assert key not in setup_battle_shared_payoff._data.keys()
        win_rate = setup_battle_shared_payoff[home, away]
        assert key in setup_battle_shared_payoff._data.keys()  # set key in ``_win_rate``
        assert len(win_rate.shape) == 1
        assert win_rate[0] == pytest.approx(0.5)  # no enough game results, return 0.5 by default

        # test players list
        for i in range(314):
            home = np.random.choice(setup_battle_shared_payoff.players)
            away = np.random.choice(setup_battle_shared_payoff.players)
            env_num = 1
            episode_num = 1
            job_result = [[random_job_result() for _ in range(env_num)] for _ in range(episode_num)]
            job_info = {
                'player_id': [home.player_id, away.player_id],
                'episode_num': episode_num,
                'env_num': env_num,
                'result': job_result
            }
            assert setup_battle_shared_payoff.update(job_info)
        for i in range(314):
            home_num = np.random.randint(1, N + 1)
            home = np.random.choice(setup_battle_shared_payoff.players, home_num).tolist()
            away_num = np.random.randint(1, N + 1)
            away = np.random.choice(setup_battle_shared_payoff.players, away_num).tolist()
            win_rates = setup_battle_shared_payoff[home, away]
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
            assert id(p.payoff) == id(setup_battle_shared_payoff)


def get_solo_active_player(config, payoff):
    return SoloActivePlayer(
        config.league.main_player, 'default', payoff, 'ckpt_{}.pth'.format('solo_default'), 'solo_default', 0
    )


@pytest.mark.unittest
class TestSoloSharedPayoff:

    def test_update(self, setup_solo_shared_payoff, random_job_result, get_job_result_categories, setup_config):
        games_per_player = 4
        player_list = [get_solo_active_player(setup_config, setup_solo_shared_payoff)]
        for p in player_list:
            setup_solo_shared_payoff.add_player(p)
        with pytest.raises(Exception):
            setup_solo_shared_payoff.add_player(player_list[0])

        for player in player_list:
            for i in range(games_per_player):
                episode_num = 2
                env_num = 4
                job_result = [[random_job_result() for _ in range(env_num)] for _ in range(episode_num)]
                job_info = {
                    'player_id': [player.player_id],
                    'episode_num': episode_num,
                    'env_num': env_num,
                    'result': job_result
                }
                key = player.player_id
                print('old:', setup_solo_shared_payoff._data[key])
                old = setup_solo_shared_payoff._data[key].copy()
                # old = setup_solo_shared_payoff._data[key]
                assert setup_solo_shared_payoff.update(job_info)

                old.append(job_info)
                assert old == setup_solo_shared_payoff._data[key]

        # test shared payoff
        for p in player_list:
            assert id(p.payoff) == id(setup_solo_shared_payoff)

    def test_getitem(self, setup_solo_shared_payoff, random_job_result, setup_config):
        games_per_player = 4
        player_list = [get_solo_active_player(setup_config, setup_solo_shared_payoff)]
        for p in player_list:
            setup_solo_shared_payoff.add_player(p)

        # test key not in setup_solo_shared_payoff._data
        player = player_list[0]
        key = player.player_id
        assert key not in setup_solo_shared_payoff._data.keys()
        result_queue = setup_solo_shared_payoff[player]
        assert key in setup_solo_shared_payoff._data.keys()
        assert len(result_queue) == 0

        # test shared payoff
        for p in player_list:
            assert id(p.payoff) == id(setup_solo_shared_payoff)

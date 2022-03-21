import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest
from easydict import EasyDict

from ding.league.player import Player
from ding.league.shared_payoff import BattleRecordDict, create_payoff
from ding.league.metric import LeagueMetricEnv

env = LeagueMetricEnv()


@pytest.mark.unittest
class TestBattleRecordDict:

    def test_init(self):
        data1 = defaultdict(BattleRecordDict)
        data1['test_player_0-test_player_1'] *= 1
        assert data1['test_player_0-test_player_1']['wins'] == 0
        assert data1['test_player_0-test_player_1']['draws'] == 0
        assert data1['test_player_0-test_player_1']['losses'] == 0
        assert data1['test_player_0-test_player_1']['games'] == 0
        with pytest.raises(KeyError):
            tmp = data1['test_player_0-test_player_1']['xxx']


@pytest.fixture(scope='function')
def setup_battle_shared_payoff():
    cfg = EasyDict({'type': 'battle', 'decay': 0.99})
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
        total_agent_step=0,
        rating=env.create_rating(),
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
                if home == away:
                    continue  # ignore self-play case
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


if __name__ == '__main__':
    pytest.main(["-sv", os.path.basename(__file__)])

import os

import numpy as np
import pytest
from easydict import EasyDict

from ding.league.player import Player, HistoricalPlayer, ActivePlayer, create_player
from ding.league.shared_payoff import create_payoff
from ding.league.starcraft_player import MainPlayer, MainExploiter, LeagueExploiter
from ding.league.tests.league_test_default_config import league_test_config
from ding.league.metric import LeagueMetricEnv

ONE_PHASE_STEP = 2000
env = LeagueMetricEnv()


@pytest.fixture(scope='function')
def setup_payoff():
    cfg = EasyDict({'type': 'battle', 'decay': 0.99})
    return create_payoff(cfg)


@pytest.fixture(scope='function')
def setup_league(setup_payoff):
    players = []
    for category in ['zerg', 'terran', 'protoss']:
        # main_player
        main_player_name = '{}_{}'.format('MainPlayer', category)
        players.append(
            create_player(
                league_test_config.league, 'main_player', league_test_config.league.main_player, category, setup_payoff,
                'ckpt_{}.pth'.format(main_player_name), main_player_name, 0, env.create_rating()
            )
        )
        # main_exloiter
        main_exploiter_name = '{}_{}'.format('MainExploiter', category)
        players.append(
            create_player(
                league_test_config.league, 'main_exploiter', league_test_config.league.main_exploiter, category,
                setup_payoff, 'ckpt_{}.pth'.format(main_exploiter_name), main_exploiter_name, 0, env.create_rating()
            )
        )
        # league_exploiter
        league_exploiter_name = '{}_{}'.format('LeagueExploiter', category)
        for i in range(2):
            players.append(
                create_player(
                    league_test_config.league,
                    'league_exploiter',
                    league_test_config.league.league_exploiter,
                    category,
                    setup_payoff,
                    'ckpt_{}.pth'.format(league_exploiter_name),
                    league_exploiter_name,
                    0,
                    env.create_rating(),
                )
            )
        # historical player: sl player is used as initial HistoricalPlayer
        sl_hp_name = '{}_{}_sl'.format('MainPlayer', category)
        players.append(
            create_player(
                league_test_config.league,
                'historical_player',
                EasyDict(),
                category,
                setup_payoff,
                'ckpt_sl_{}'.format(sl_hp_name),
                sl_hp_name,
                0,
                env.create_rating(),
                parent_id=main_player_name,
            )
        )
    for p in players:
        setup_payoff.add_player(p)
    return players


@pytest.mark.unittest
class TestMainPlayer:

    def test_get_job(self, setup_league, setup_payoff):
        N = 10
        # no indicated p
        # test get_job
        for p in setup_league:
            if isinstance(p, MainPlayer):
                for i in range(N):
                    job_dict = p.get_job()
                    assert isinstance(job_dict, dict)
                    opponent = job_dict['opponent']
                    assert isinstance(opponent, Player)
                    assert opponent in setup_league

        # payoff = setup_league[np.random.randint(0, len(setup_league))].payoff  # random select reference
        hp_list = []
        for p in setup_league:
            if isinstance(p, ActivePlayer):
                p.total_agent_step = 2 * ONE_PHASE_STEP
                hp = p.snapshot(env)
                hp_list.append(hp)
                setup_payoff.add_player(hp)
        setup_league += hp_list  # 12+3 + 12

        # test get_job with branch prob
        pfsp, sp, veri = False, False, False
        for p in setup_league:
            if isinstance(p, MainPlayer):
                while True:
                    job_dict = p.get_job()
                    opponent = job_dict['opponent']
                    if isinstance(opponent, HistoricalPlayer) and 'MainPlayer' in opponent.parent_id:
                        veri = True
                    elif isinstance(opponent, HistoricalPlayer):
                        pfsp = True
                    elif isinstance(opponent, MainPlayer):
                        sp = True
                    else:
                        raise Exception("Main Player selects a wrong opponent {}", type(opponent))
                    if veri and pfsp and sp:
                        break

    def test_snapshot(self, setup_league, setup_payoff):
        N = 10
        for p in setup_league:
            for i in range(N):
                if isinstance(p, ActivePlayer):
                    hp = p.snapshot(env)
                    assert isinstance(hp, HistoricalPlayer)
                    assert id(hp.payoff) == id(p.payoff)
                    assert hp.parent_id == p.player_id

    def test_is_trained_enough(self, setup_league, setup_payoff):
        for p in setup_league:
            if isinstance(p, ActivePlayer):
                assert not p.is_trained_enough()
                assert p._last_enough_step == 0
                # step_passed < ONE_PHASE_STEP
                p.total_agent_step = ONE_PHASE_STEP * 0.99
                assert not p.is_trained_enough()
                assert p._last_enough_step == 0
                # ONE_PHASE_STEP < step_passed < 2*ONE_PHASE_STEP, but low win rate
                p.total_agent_step = ONE_PHASE_STEP + 1
                assert not p.is_trained_enough()
                assert p._last_enough_step == 0

        # prepare HistoricalPlayer
        # payoff = setup_league[np.random.randint(0, len(setup_league))].payoff  # random select reference
        hp_list = []
        for p in setup_league:
            if isinstance(p, MainPlayer):
                hp = p.snapshot(env)
                setup_payoff.add_player(hp)
                hp_list.append(hp)
        setup_league += hp_list

        # update 10 wins against all historical players, should be trained enough
        N = 10
        assert isinstance(setup_league[0], MainPlayer)
        for n in range(N):
            for hp in [p for p in setup_league if isinstance(p, HistoricalPlayer)]:
                match_info = {
                    'player_id': [setup_league[0].player_id, hp.player_id],
                    'result': [['wins']],
                }
                result = setup_payoff.update(match_info)
                assert result
        assert setup_league[0]._total_agent_step > ONE_PHASE_STEP
        assert setup_league[0]._last_enough_step == 0
        assert setup_league[0]._last_enough_step != setup_league[0]._total_agent_step
        assert setup_league[0].is_trained_enough()
        assert setup_league[0]._last_enough_step == setup_league[0]._total_agent_step

        # update 10 draws against all historical players, should be not trained enough;
        # then update ``total_agent_step`` to 2*ONE_PHASE_STEP, should be trained enough
        assert isinstance(setup_league[5], MainPlayer)
        for n in range(N):
            for hp in hp_list:
                match_info = {
                    'player_id': [setup_league[5].player_id, hp.player_id],
                    'result': [['draws']],
                }
                result = setup_payoff.update(match_info)
                assert result
        assert setup_league[5]._total_agent_step > ONE_PHASE_STEP
        assert not setup_league[5].is_trained_enough()
        setup_league[5].total_agent_step = 2 * ONE_PHASE_STEP
        assert setup_league[5].is_trained_enough()

    def test_mutate(self, setup_league, setup_payoff):
        # main players do not mutate
        assert isinstance(setup_league[0], MainPlayer)
        for _ in range(10):
            assert setup_league[0].mutate({}) is None

    def test_sp_historical(self, setup_league, setup_payoff):
        N = 10
        main1 = setup_league[0]  # 'zerg'
        main2 = setup_league[5]  # 'terran'
        assert isinstance(main1, MainPlayer)
        assert isinstance(main2, MainPlayer)
        for n in range(N):
            match_info = {
                'player_id': [main1.player_id, main2.player_id],
                'result': [['wins']],
            }
            result = setup_payoff.update(match_info)
            assert result
        for _ in range(200):
            opponent = main2._sp_branch()
            condition1 = opponent.category == 'terran' or opponent.category == 'protoss'
            # condition2 means: zerg_main_opponent is too strong, so that must choose a historical weaker one
            condition2 = opponent.category == 'zerg' and isinstance(
                opponent, HistoricalPlayer
            ) and opponent.parent_id == main1.player_id
            assert condition1 or condition2, (condition1, condition2)


@pytest.mark.unittest
class TestMainExploiter:

    def test_get_job(self, setup_league, random_job_result, setup_payoff):
        assert isinstance(setup_league[1], MainExploiter)
        job_dict = setup_league[1].get_job()
        opponent = job_dict['opponent']
        assert isinstance(opponent, MainPlayer)

        N = 10
        # payoff = setup_league[np.random.randint(0, len(setup_league))].payoff  # random select reference
        for n in range(N):
            for p in setup_league:
                if isinstance(p, MainPlayer):
                    match_info = {
                        'player_id': [setup_league[1].player_id, p.player_id],
                        'result': [['losses']],
                    }
                    assert setup_payoff.update(match_info)

        job_dict = setup_league[1].get_job()
        opponent = job_dict['opponent']
        # as long as main player, both active and historical are ok
        assert (isinstance(opponent, HistoricalPlayer)
                and 'MainPlayer' in opponent.parent_id) or isinstance(opponent, MainPlayer)
        hp_list = []
        for i in range(3):
            for p in setup_league:
                if isinstance(p, MainPlayer):
                    p.total_agent_step = (i + 1) * 2 * ONE_PHASE_STEP
                    hp = p.snapshot(env)
                    setup_payoff.add_player(hp)
                    hp_list.append(hp)
        setup_league += hp_list

        no_main_player_league = [p for p in setup_league if not isinstance(p, MainPlayer)]
        for i in range(10000):
            home = np.random.choice(no_main_player_league)
            away = np.random.choice(no_main_player_league)
            result = random_job_result()
            match_info = {
                'player_id': [home.player_id, away.player_id],
                'result': [[result]],
            }
            assert setup_payoff.update(match_info)

        for i in range(10):
            job_dict = setup_league[1].get_job()
            opponent = job_dict['opponent']
            # as long as main player, both active and historical are ok
            assert (isinstance(opponent, HistoricalPlayer)
                    and 'MainPlayer' in opponent.parent_id) or isinstance(opponent, MainPlayer)

    def test_is_trained_enough(self, setup_league):
        # only a few differences from `is_trained_enough` of MainPlayer
        pass

    def test_mutate(self, setup_league):
        assert isinstance(setup_league[1], MainExploiter)
        info = {'reset_checkpoint_path': 'pretrain_checkpoint.pth'}
        for _ in range(10):
            assert setup_league[1].mutate(info) == info['reset_checkpoint_path']


@pytest.mark.unittest
class TestLeagueExploiter:

    def test_get_job(self, setup_league):
        assert isinstance(setup_league[2], LeagueExploiter)
        job_dict = setup_league[2].get_job()
        opponent = job_dict['opponent']
        assert isinstance(opponent, HistoricalPlayer)
        assert isinstance(setup_league[3], LeagueExploiter)
        job_dict = setup_league[3].get_job()
        opponent = job_dict['opponent']
        assert isinstance(opponent, HistoricalPlayer)

    def test_is_trained_enough(self, setup_league):
        # this function is the same as `is_trained_enough` of MainPlayer
        pass

    def test_mutate(self, setup_league):
        assert isinstance(setup_league[2], LeagueExploiter)
        info = {'reset_checkpoint_path': 'pretrain_checkpoint.pth'}
        results = []
        for _ in range(1000):
            results.append(setup_league[2].mutate(info))
        freq = len([t for t in results if t]) * 1.0 / len(results)
        assert 0.2 <= freq <= 0.3  # approximate


if __name__ == '__main__':
    pytest.main(["-sv", os.path.basename(__file__)])

import pytest
import numpy as np
from nervex.league.shared_payoff import SharedPayoff
from nervex.league.player import Player, MainPlayer, MainExploiter, LeagueExploiter, HistoricalPlayer, ActivePlayer

STRONG = 0.7
ONE_PHASE_STEPS = 2e3
MIN_VALID = 0.2


@pytest.fixture(scope='function')
def setup_payoff():
    return SharedPayoff(decay=0.99)


@pytest.fixture(scope='function')
def setup_league(setup_payoff):
    players = []
    main_player_branch = {'pfsp': 0.5, 'sp': 0.35, 'verification': 0.15}
    main_exploiter_branch = {'main_players': 1.0}
    league_exploiter_branch = {'pfsp': 1.0}
    for race in ['zerg', 'terran', 'protoss']:
        main_player_name = '{}_{}'.format('MainPlayer', race)
        players.append(
            MainPlayer(
                race,
                setup_payoff,
                'ckpt_{}.pth'.format(main_player_name),
                main_player_name,
                branch_probs=main_player_branch,
                strong_win_rate=STRONG,
                one_phase_steps=ONE_PHASE_STEPS
            )
        )

        main_exploiter_name = '{}_{}'.format('MainExploiter', race)
        players.append(
            MainExploiter(
                race,
                setup_payoff,
                'ckpt_{}.pth'.format(main_exploiter_name),
                main_exploiter_name,
                branch_probs=main_exploiter_branch,
                strong_win_rate=STRONG,
                one_phase_steps=ONE_PHASE_STEPS,
                min_valid_win_rate=MIN_VALID
            )
        )

        league_exploiter_name = '{}_{}'.format('LeagueExploiter', race)
        for i in range(2):
            players.append(
                LeagueExploiter(
                    race,
                    setup_payoff,
                    'ckpt_{}.pth'.format(league_exploiter_name),
                    league_exploiter_name,
                    branch_probs=league_exploiter_branch,
                    strong_win_rate=STRONG,
                    one_phase_steps=ONE_PHASE_STEPS
                )
            )
        # sl player is used as initial HistoricalPlayer
        sl_hp_name = '{}_{}_sl'.format('MainPlayer', race)
        players.append(
            HistoricalPlayer(
                race, setup_payoff, 'ckpt_sl_{}'.format(sl_hp_name), sl_hp_name, parent_id=main_player_name
            )
        )

    for p in players:
        setup_payoff.add_player(p)

    return players


@pytest.mark.unittest
class TestMainPlayer:
    def test_get_task(self, setup_league):
        N = 10
        # no indicated p
        for p in setup_league:
            if isinstance(p, MainPlayer):
                for i in range(N):
                    opponent = p.get_task()
                    assert isinstance(opponent, Player)
                    assert opponent in setup_league

        payoff = setup_league[np.random.randint(0, len(setup_league))].payoff  # random select reference
        hp_list = []
        for p in setup_league:
            if isinstance(p, ActivePlayer):
                p.update_agent_step(2 * ONE_PHASE_STEPS)
                hp = p.snapshot()
                hp_list.append(hp)
                payoff.add_player(hp)
        setup_league += hp_list

        for p in setup_league:
            if isinstance(p, MainPlayer):
                for i in range(N):
                    for idx, prob in enumerate([0.4, 0.6, 0.9]):
                        opponent = p.get_task(p=prob)
                        if idx == 0:
                            assert isinstance(opponent, HistoricalPlayer)
                        elif idx == 1:
                            assert isinstance(opponent, MainPlayer)
                        else:
                            assert isinstance(opponent, HistoricalPlayer) and 'MainPlayer' in opponent.parent_id

    def test_update_agent_step(self, setup_league):
        assert setup_league[0]._total_agent_steps == 0
        setup_league[0].update_agent_step(ONE_PHASE_STEPS)
        assert setup_league[0]._total_agent_steps == ONE_PHASE_STEPS

    def test_snapshot(self, setup_league):
        N = 10
        for p in setup_league:
            for i in range(N):
                if isinstance(p, ActivePlayer):
                    hp = p.snapshot()
                    assert isinstance(hp, HistoricalPlayer)
                    assert id(hp.payoff) == id(p.payoff)
                    assert hp.parent_id == p.player_id

    def test_is_trained_enough(self, setup_league):
        for p in setup_league:
            if isinstance(p, ActivePlayer):
                assert not p.is_trained_enough()
                assert p._last_enough_steps == 0

                p.update_agent_step(ONE_PHASE_STEPS * 0.99)
                assert not p.is_trained_enough()
                assert p._last_enough_steps == 0

                p.update_agent_step(ONE_PHASE_STEPS + 1)
                assert not p.is_trained_enough()
                assert p._last_enough_steps == 0

        payoff = setup_league[np.random.randint(0, len(setup_league))].payoff  # random select reference
        # prepare HistoricalPlayer
        hp_list = []
        for p in setup_league:
            if isinstance(p, MainPlayer):
                hp = p.snapshot()
                payoff.add_player(hp)
                hp_list.append(hp)
        setup_league += hp_list

        N = 10
        assert isinstance(setup_league[0], MainPlayer)
        for n in range(N):
            for hp in [p for p in setup_league if isinstance(p, HistoricalPlayer)]:
                match_info = {
                    'home_id': setup_league[0].player_id,
                    'away_id': hp.player_id,
                    'result': 'wins',
                }
                result = payoff.update(match_info)
                assert result

        assert isinstance(setup_league[5], MainPlayer)
        for n in range(N):
            for hp in hp_list:
                match_info = {
                    'home_id': setup_league[5].player_id,
                    'away_id': hp.player_id,
                    'result': 'draws',
                }
                result = payoff.update(match_info)
                assert result

        assert setup_league[0]._total_agent_steps > ONE_PHASE_STEPS
        assert setup_league[0]._last_enough_steps == 0
        assert setup_league[0]._last_enough_steps != setup_league[0]._total_agent_steps
        assert setup_league[0].is_trained_enough()
        assert setup_league[0]._last_enough_steps == setup_league[0]._total_agent_steps

        assert setup_league[5]._total_agent_steps > ONE_PHASE_STEPS
        assert not setup_league[5].is_trained_enough()

        setup_league[5].update_agent_step(2 * ONE_PHASE_STEPS)
        assert setup_league[5].is_trained_enough()

    def test_mutate(self, setup_league):
        assert isinstance(setup_league[0], MainPlayer)
        for _ in range(10):
            assert setup_league[0].mutate({}) is None


@pytest.mark.unittest
class TestMainExploiter:
    def test_get_task(self, setup_league, random_match_result):
        assert isinstance(setup_league[1], MainExploiter)
        opponent = setup_league[1].get_task()
        assert isinstance(opponent, MainPlayer)

        N = 10
        payoff = setup_league[np.random.randint(0, len(setup_league))].payoff  # random select reference
        for n in range(N):
            for p in setup_league:
                if isinstance(p, MainPlayer):
                    match_info = {'home_id': setup_league[1].player_id, 'away_id': p.player_id, 'result': 'losses'}
                    assert payoff.update(match_info)

        opponent = setup_league[1].get_task()
        assert isinstance(opponent, HistoricalPlayer) and 'MainPlayer' in opponent.parent_id
        hp_list = []
        for i in range(3):
            for p in setup_league:
                if isinstance(p, MainPlayer):
                    p.update_agent_step((i + 1) * 2 * ONE_PHASE_STEPS)
                    hp = p.snapshot()
                    payoff.add_player(hp)
                    hp_list.append(hp)
        setup_league += hp_list

        no_main_player_league = [p for p in setup_league if not isinstance(p, MainPlayer)]
        for i in range(10000):
            home = np.random.choice(no_main_player_league)
            away = np.random.choice(no_main_player_league)
            result = random_match_result()
            match_info = {'home_id': home.player_id, 'away_id': away.player_id, 'result': result}
            assert payoff.update(match_info)

        for i in range(10):
            opponent = setup_league[1].get_task()
            assert isinstance(opponent, HistoricalPlayer) and 'MainPlayer' in opponent.parent_id

    def test_is_trained_enough(self, setup_league):
        # only a few differences from `is_trained_enough` of MainPlayer
        pass

    def test_mutate(self, setup_league):
        assert isinstance(setup_league[1], MainExploiter)
        info = {'pretrain_checkpoint_path': 'pretrain_checkpoint.pth'}
        for _ in range(10):
            assert setup_league[1].mutate(info) == info['pretrain_checkpoint_path']


@pytest.mark.unittest
class TestLeagueExploiter:
    def test_get_task(self, setup_league):
        pass

    def test_is_trained_enough(self, setup_league):
        # this function is the same as `is_trained_enough` of MainPlayer
        pass

    def test_mutate(self, setup_league):
        assert isinstance(setup_league[2], LeagueExploiter)
        info = {'pretrain_checkpoint_path': 'pretrain_checkpoint.pth'}
        results = []
        for _ in range(1000):
            results.append(setup_league[2].mutate(info))
        freq = len([t for t in results if t]) * 1.0 / len(results)
        assert freq >= 0.2 and freq <= 0.3  # approximate

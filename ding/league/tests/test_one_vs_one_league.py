import os
import random

import pytest
import copy
from easydict import EasyDict
import torch

from ding.league import create_league

one_vs_one_league_default_config = dict(
    league=dict(
        league_type='one_vs_one',
        import_names=["ding.league"],
        # ---player----
        # "player_category" is just a name. Depends on the env.
        # For example, in StarCraft, this can be ['zerg', 'terran', 'protoss'].
        player_category=['default'],
        # Support different types of active players for solo and battle league.
        # For solo league, supports ['solo_active_player'].
        # For battle league, supports ['battle_active_player', 'main_player', 'main_exploiter', 'league_exploiter'].
        active_players=dict(
            naive_sp_player=1,  # {player_type: player_num}
        ),
        naive_sp_player=dict(
            # There should be keys ['one_phase_step', 'branch_probs', 'strong_win_rate'].
            # Specifically for 'main_exploiter' of StarCraft, there should be an additional key ['min_valid_win_rate'].
            one_phase_step=10,
            branch_probs=dict(
                pfsp=0.5,
                sp=0.5,
            ),
            strong_win_rate=0.7,
        ),
        # "use_pretrain" means whether to use pretrain model to initialize active player.
        use_pretrain=False,
        # "use_pretrain_init_historical" means whether to use pretrain model to initialize historical player.
        # "pretrain_checkpoint_path" is the pretrain checkpoint path used in "use_pretrain" and
        # "use_pretrain_init_historical". If both are False, "pretrain_checkpoint_path" can be omitted as well.
        # Otherwise, "pretrain_checkpoint_path" should list paths of all player categories.
        use_pretrain_init_historical=False,
        pretrain_checkpoint_path=dict(default='default_cate_pretrain.pth', ),
        # ---payoff---
        payoff=dict(
            # Supports ['battle']
            type='battle',
            decay=0.99,
            min_win_rate_games=8,
        ),
        path_policy='./league',
    ),
)
one_vs_one_league_default_config = EasyDict(one_vs_one_league_default_config)


def get_random_result():
    ran = random.random()
    if ran < 1. / 3:
        return "wins"
    elif ran < 1. / 2:
        return "losses"
    else:
        return "draws"


@pytest.mark.unittest
class TestOneVsOneLeague:

    def test_naive(self):
        league = create_league(one_vs_one_league_default_config.league)
        assert (len(league.active_players) == 1)
        assert (len(league.historical_players) == 0)
        active_player_ids = [p.player_id for p in league.active_players]
        assert set(active_player_ids) == set(league.active_players_ids)
        active_player_id = active_player_ids[0]

        active_player_ckpt = league.active_players[0].checkpoint_path
        tmp = torch.tensor([1, 2, 3])
        path_policy = one_vs_one_league_default_config.league.path_policy
        torch.save(tmp, active_player_ckpt)

        # judge_snapshot & update_active_player
        assert not league.judge_snapshot(active_player_id)
        player_update_dict = {
            'player_id': active_player_id,
            'train_iteration': one_vs_one_league_default_config.league.naive_sp_player.one_phase_step * 2,
        }
        league.update_active_player(player_update_dict)
        assert league.judge_snapshot(active_player_id)
        historical_player_ids = [p.player_id for p in league.historical_players]
        assert len(historical_player_ids) == 1
        historical_player_id = historical_player_ids[0]

        # get_job_info, eval_flag=False
        vs_active = False
        vs_historical = False
        while True:
            collect_job_info = league.get_job_info(active_player_id, eval_flag=False)
            assert collect_job_info['agent_num'] == 2
            assert len(collect_job_info['checkpoint_path']) == 2
            assert collect_job_info['launch_player'] == active_player_id
            assert collect_job_info['player_id'][0] == active_player_id
            if collect_job_info['player_active_flag'][1]:
                assert collect_job_info['player_id'][1] == collect_job_info['player_id'][0]
                vs_active = True
            else:
                assert collect_job_info['player_id'][1] == historical_player_id
                vs_historical = True
            if vs_active and vs_historical:
                break

        # get_job_info, eval_flag=False
        eval_job_info = league.get_job_info(active_player_id, eval_flag=True)
        assert eval_job_info['agent_num'] == 1
        assert len(eval_job_info['checkpoint_path']) == 1
        assert eval_job_info['launch_player'] == active_player_id
        assert eval_job_info['player_id'][0] == active_player_id
        assert len(eval_job_info['player_id']) == 1
        assert len(eval_job_info['player_active_flag']) == 1
        assert eval_job_info['eval_opponent'] in league.active_players[0]._eval_opponent_difficulty

        # finish_job

        episode_num = 5
        env_num = 8
        player_id = [active_player_id, historical_player_id]
        result = [[get_random_result() for __ in range(8)] for _ in range(5)]
        payoff_update_info = {
            'launch_player': active_player_id,
            'player_id': player_id,
            'episode_num': episode_num,
            'env_num': env_num,
            'result': result,
        }
        league.finish_job(payoff_update_info)
        wins = 0
        games = episode_num * env_num
        for i in result:
            for j in i:
                if j == 'wins':
                    wins += 1
        league.payoff[league.active_players[0], league.historical_players[0]] == wins / games

        os.popen("rm -rf {}".format(path_policy))
        print("Finish!")

    def test_league_info(self):
        cfg = copy.deepcopy(one_vs_one_league_default_config.league)
        cfg.path_policy = 'test_league_info'
        league = create_league(cfg)
        active_player_id = [p.player_id for p in league.active_players][0]
        active_player_ckpt = [p.checkpoint_path for p in league.active_players][0]
        tmp = torch.tensor([1, 2, 3])
        torch.save(tmp, active_player_ckpt)
        assert (len(league.active_players) == 1)
        assert (len(league.historical_players) == 0)
        print('\n')
        print(repr(league.payoff))
        print(league.player_rank(string=True))
        league.judge_snapshot(active_player_id, force=True)
        for i in range(10):
            job = league.get_job_info(active_player_id, eval_flag=False)
            payoff_update_info = {
                'launch_player': active_player_id,
                'player_id': job['player_id'],
                'episode_num': 2,
                'env_num': 4,
                'result': [[get_random_result() for __ in range(4)] for _ in range(2)]
            }
            league.finish_job(payoff_update_info)
            # if not self-play
            if job['player_id'][0] != job['player_id'][1]:
                win_loss_result = sum(payoff_update_info['result'], [])
                home = league.get_player_by_id(job['player_id'][0])
                away = league.get_player_by_id(job['player_id'][1])
                home.rating, away.rating = league.metric_env.rate_1vs1(home.rating, away.rating, win_loss_result)
        print(repr(league.payoff))
        print(league.player_rank(string=True))
        os.popen("rm -rf {}".format(cfg.path_policy))


if __name__ == '__main__':
    pytest.main(["-sv", os.path.basename(__file__)])

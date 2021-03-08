import os
import numpy as np
import pytest
from easydict import EasyDict
import random
import torch

from nervex.league import BaseLeague, register_league, create_league, ActivePlayer
from nervex.utils import deep_merge_dicts
from nervex.config import one_vs_one_league_default_config


class TestOneVsOneLeague:

    @pytest.mark.unittest
    def test_naive(self):
        league = create_league(one_vs_one_league_default_config.league)
        assert (len(league.active_players) == 1)
        assert (len(league.historical_players) == 0)
        active_player_ids = [p.player_id for p in league.active_players]
        assert set(active_player_ids) == set(league.active_players_ids)
        active_player_id = active_player_ids[0]

        active_player_ckpt = league.active_players[0].checkpoint_path
        tmp = torch.tensor([1, 2, 3])
        torch.save(tmp, active_player_ckpt)

        # judge_snapshot & update_active_player
        assert not league.judge_snapshot(active_player_id)
        player_update_dict = {
            'player_id': active_player_id,
            'train_step': one_vs_one_league_default_config.league.naive_sp_player.one_phase_step * 2,
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
        def get_random_result():
            ran = random.random()
            if ran < 1. / 3:
                return "wins"
            elif ran < 2. / 3:
                return "losses"
            else:
                return "draws"

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

        os.popen("rm -rf naive_sp_player*")

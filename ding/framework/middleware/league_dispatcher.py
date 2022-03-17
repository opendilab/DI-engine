import torch
import time
from typing import TYPE_CHECKING

from ding.league import player
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.utils import DistributedWriter
    from ding.league.base_league import BaseLeague


def league_dispatcher(task: "Task", cfg: dict, tb_logger: "DistributedWriter", league: "BaseLeague", policies: dict):

    def update_active_player(player_info):
        league.update_active_player(player_info)
        league.judge_snapshot(player_info["player_id"])

    task.on("update_active_player", update_active_player)

    # Wait for all players online
    online_learners = {}
    for player_id in league.active_players_ids:
        online_learners[player_id] = False

    def learner_online(player_id):
        print("Get learner", player_id)
        online_learners[player_id] = True

    task.on("learner_online", learner_online)

    def _league(ctx: "Context"):
        print("Waiting for all learners online")
        while True:
            if all(online_learners.values()):
                break
            time.sleep(0.1)
        print("League dispatching on node {}".format(task.router.node_id))
        # One episode each round
        i = ctx.total_step % len(league.active_players_ids)
        player_id, player_ckpt_path = league.active_players_ids[i], league.active_players_ckpts[i]

        job = league.get_job_info(player_id)
        opponent_player_id = job['player_id'][1]

        if 'historical' in opponent_player_id:
            opponent_policy = policies['historical'].collect_mode
            opponent_path = job['checkpoint_path'][1]
            opponent_policy.load_state_dict(torch.load(opponent_path, map_location='cpu'))
        else:
            opponent_policy = policies[opponent_player_id].collect_mode
        # Watch out that in parallel mode, we should not send functions between processes, instead,
        # we should send the objects needed by the policies.
        collect_session = {
            "policies": [policies[player_id].collect_mode, opponent_policy],
            "player_id": player_id,
            "player_ckpt_path": player_ckpt_path
        }
        print("Player ID", collect_session["player_id"])

        task.emit("set_collect_session", collect_session, only_local=True)

        yield

        job_finish_info = {
            'eval_flag': True,
            'launch_player': job['launch_player'],
            'player_id': job['player_id'],
            'result': [e['result'] for e in ctx.episode_info],
        }

        league.finish_job(job_finish_info)

    return _league

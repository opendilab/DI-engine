from collections import defaultdict
from time import sleep
from threading import Lock
from dataclasses import dataclass
from typing import TYPE_CHECKING
from ding.framework import task, EventEnum
import logging

if TYPE_CHECKING:
    from easydict import EasyDict
    from ding.framework import Task, Context
    from ding.league.v2 import BaseLeague
    from ding.league.player import PlayerMeta
    from ding.league.v2.base_league import Job


class LeagueCoordinator:

    def __init__(self, cfg: "EasyDict", league: "BaseLeague") -> None:
        self.league = league
        self._lock = Lock()
        self._total_send_jobs = 0
        self._eval_frequency = 10

        task.on(EventEnum.ACTOR_GREETING, self._on_actor_greeting)
        task.on(EventEnum.LEARNER_SEND_META, self._on_learner_meta)
        task.on(EventEnum.ACTOR_FINISH_JOB, self._on_actor_job)

    def _on_actor_greeting(self, actor_id):
        print('coordinator recieve actor {} greeting\n'.format(actor_id), flush=True)
        with self._lock:
            player_num = len(self.league.active_players_ids)
            player_id = self.league.active_players_ids[self._total_send_jobs % player_num]
            job = self.league.get_job_info(player_id)
            job.job_no = self._total_send_jobs
            self._total_send_jobs += 1
        if job.job_no > 0 and job.job_no % self._eval_frequency == 0:
            job.is_eval = True
        job.actor_id = actor_id
        print('coordinator emit job (main_player: {}) to actor {}\n'.format(job.launch_player, actor_id), flush=True)
        task.emit(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=actor_id), job)

    def _on_learner_meta(self, player_meta: "PlayerMeta"):
        print('coordinator recieve learner meta for player{}\n'.format(player_meta.player_id), flush=True)
        self.league.update_active_player(player_meta)
        self.league.create_historical_player(player_meta)

    def _on_actor_job(self, job: "Job"):
        print('coordinator recieve actor finished job, palyer {}\n'.format(job.launch_player), flush=True)
        print("on_actor_job {}\n".format(job.launch_player))  # right
        self.league.update_payoff(job)

    def __del__(self):
        print('task finished, coordinator {} closed'.format(task.router.node_id), flush=True)

    def __call__(self, ctx: "Context") -> None:
        sleep(1)

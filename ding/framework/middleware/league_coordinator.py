from time import sleep
from threading import Lock
from dataclasses import dataclass
from typing import TYPE_CHECKING
from ding.framework import task, EventEnum

if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.league import BaseLeague


@dataclass
class Job:
    job_no: int = -1
    player_id: str = ""
    is_eval: bool = False


class LeagueCoordinator:

    def __init__(self, league: "BaseLeague") -> None:
        self.league = league
        self._job_iter = self._job_dispatcher()
        self._lock = Lock()
        task.on(EventEnum.ACTOR_GREETING.get_event(), self._on_actor_greeting)
        task.on(EventEnum.LEARNER_SEND_META.get_event(), self._on_learner_meta)
        task.on(EventEnum.ACTOR_FINISH_JOB, self._on_actor_job)

    def _on_actor_greeting(self, actor_id):
        with self._lock:
            job: "Job" = next(self._job_iter)
        if job.job_no > 0 and job.job_no % 10 == 0:  # 1/10 turn job into eval mode
            job.is_eval = True
        job.actor_id = actor_id
        task.emit(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.get_event(actor_id), job)

    def _on_actor_job(self, job: "Job"):
        self.league.update_payoff(job)

    def _on_learner_meta(self, player_meta: "PlayerMeta"):
        self.league.update_active_player(player_meta)
        self.league.create_historical_player(player_meta)

    def __call__(self, ctx: "Context") -> None:
        sleep(0.1)

    def _job_dispatcher(self) -> "Job":
        i = 0
        while True:
            player_num = len(self.league.active_players_ids)
            player_id = self.league.active_players_ids[i % player_num]
            job = self.league.get_job_info(player_id)
            job.job_no = i
            i += 1
            yield job

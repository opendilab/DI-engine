import pytest
import time
from unittest.mock import patch
from ding.framework import task, Parallel
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import LeagueCoordinator
from ding.league.v2 import BaseLeague, Job
from ding.framework import EventEnum


class MockLeague:

    def __init__(self):
        self.active_players_ids = ["player_0", "player_1", "player_2"]
        self.update_payoff_cnt = 0
        self.update_active_player_cnt = 0
        self.create_historical_player_cnt = 0
        self.get_job_info_cnt = 0

    def update_payoff(self, job):
        self.update_payoff_cnt += 1

    def update_active_player(self, meta):
        self.update_active_player_cnt += 1

    def create_historical_player(self, meta):
        self.create_historical_player_cnt += 1

    def get_job_info(self, player_id):
        self.get_job_info_cnt += 1
        return Job(launch_player=player_id, players=[])


def _main():
    with task.start():
        if task.router.node_id == 0:
            with patch("ding.league.BaseLeague", MockLeague):
                league = MockLeague()
                coordinator = LeagueCoordinator(league)
                time.sleep(3)
                assert league.update_payoff_cnt == 3
                assert league.update_active_player_cnt == 3
                assert league.create_historical_player_cnt == 3
                assert league.get_job_info_cnt == 3
        elif task.router.node_id == 1:
            # test ACTOR_GREETING
            res = []
            task.on(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=task.router.node_id),
                lambda job: res.append(job))
            for _ in range(3):
                task.emit(EventEnum.ACTOR_GREETING, task.router.node_id)
            time.sleep(3)
            assert task.router.node_id == res[-1].actor_id
        elif task.router.node_id == 2:
            # test LEARNER_SEND_META
            for _ in range(3):
                task.emit(EventEnum.LEARNER_SEND_META, {"meta": task.router.node_id})
            time.sleep(3)
        elif task.router.node_id == 3:
            # test ACTOR_FINISH_JOB
            job = Job(-1, task.router.node_id, False)
            for _ in range(3):
                task.emit(EventEnum.ACTOR_FINISH_JOB, job)
            time.sleep(3)
        else:
            raise Exception("Invalid node id {}".format(task.router.is_active)) 


@pytest.mark.unittest
def test_coordinator():
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="star")(_main)


if __name__ == "__main__":
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="star")(_main)

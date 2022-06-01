from time import sleep
import pytest
from copy import deepcopy
from ding.envs import BaseEnvManager
from ding.framework.middleware.tests.league_config import cfg
from ding.framework.middleware import LeagueActor, LeagueCoordinator
from ding.league.player import PlayerMeta
from ding.framework.storage import FileStorage

from ding.framework.task import task, Parallel
from ding.league.v2.base_league import Job
from ding.model import VAC
from ding.policy.ppo import PPOPolicy
from dizoo.league_demo.game_env import GameEnv

from unittest.mock import patch
import random

def prepare_test():
    global cfg
    cfg = deepcopy(cfg)

    def env_fn():
        env = BaseEnvManager(
            env_fn=[lambda: GameEnv(cfg.env.env_type) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    def policy_fn():
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        return policy

    return cfg, env_fn, policy_fn

class MockLeague:

    def __init__(self):
        self.active_players_ids = ["main_player_default_0", "main_player_default_1", "main_player_default_2"]
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
        other_players = [i for i in self.active_players_ids if i != player_id]
        another_palyer = random.choice(other_players)
        return Job(
            launch_player=player_id, 
            players=[
                PlayerMeta(player_id=player_id, checkpoint=FileStorage(path=None), total_agent_step=0),
                PlayerMeta(player_id=another_palyer, checkpoint=FileStorage(path=None), total_agent_step=0)
            ]
        )

N_ACTORS = 5

def _main():
    cfg, env_fn, policy_fn = prepare_test()

    with task.start(async_mode=True):
        if task.router.node_id == 0:
            league = MockLeague()
            coordinator = LeagueCoordinator(league)
            sleep(2)
            with patch("ding.league.BaseLeague", MockLeague):
                task.use(coordinator)
            sleep(15)
            # print(league.get_job_info_cnt)
            assert league.get_job_info_cnt == N_ACTORS
            assert league.update_payoff_cnt == N_ACTORS
        else:
            task.use(LeagueActor(cfg, env_fn, policy_fn))

        task.run(max_step=1)


@pytest.mark.unittest
def test_league_actor():
    Parallel.runner(n_parallel_workers=N_ACTORS+1, protocol="tcp", topology="mesh")(_main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=N_ACTORS+1, protocol="tcp", topology="mesh")(_main)
# replicas = 10
# un parallel worker 改成1
# from time import sleep
# from boto import config
# import pytest
# from copy import deepcopy

# from ding.framework.middleware.league_learner import LeagueLearner
# from ding.framework.middleware.tests.league_config import cfg
# from ding.framework.middleware import LeagueActor, LeagueCoordinator
# from ding.league.player import PlayerMeta, create_player
# from ding.league.v2 import BaseLeague
# from ding.framework.storage import FileStorage

# from ding.framework.task import task, Parallel
# from ding.league.v2.base_league import Job
# from ding.model import VAC
# from ding.policy.ppo import PPOPolicy
# from dizoo.league_demo.game_env import GameEnv

from copy import deepcopy
from time import sleep
import torch
import pytest
import random

from ding.envs import BaseEnvManager
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.framework import EventEnum
from ding.framework.task import task, Parallel
from ding.framework.middleware import LeagueCoordinator, LeagueActor, LeagueLearner
from ding.framework.middleware.functional.actor_data import ActorData
from ding.framework.middleware.tests import cfg, MockLeague, MockLogger
from dizoo.league_demo.game_env import GameEnv


N_ACTORS = 2
N_LEARNERS = 2

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

def _main():
    cfg, env_fn, policy_fn = prepare_test()
    league = MockLeague(cfg.policy.other.league)

    with task.start(async_mode=True):
        if task.router.node_id == 0:
            task.use(LeagueCoordinator(league))
        elif task.router.node_id <= N_ACTORS:
            task.use(LeagueActor(cfg, env_fn, policy_fn))
        else:
            n_players = len(league.active_players_ids)
            player = league.active_players[task.router.node_id % n_players]
            learner = LeagueLearner(cfg, policy_fn, player)
            learner._learner._tb_logger = MockLogger()
            task.use(learner)

        task.run(max_step=10)


@pytest.mark.unittest
def test_league_actor():
    Parallel.runner(n_parallel_workers=N_ACTORS + N_LEARNERS + 1, protocol="tcp", topology="mesh")(_main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=N_ACTORS + N_LEARNERS + 1, protocol="tcp", topology="mesh")(_main)
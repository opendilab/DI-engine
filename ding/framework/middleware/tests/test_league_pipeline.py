from copy import deepcopy
from time import sleep
import pytest
from ding.envs import BaseEnvManager, EnvSupervisor
from ding.framework.context import BattleContext
from ding.framework.middleware import StepLeagueActor, LeagueCoordinator, LeagueLearner
from ding.framework.supervisor import ChildType

from ding.model import VAC
from ding.framework.task import task, Parallel
from ding.framework.middleware.tests import cfg, MockLeague, MockLogger
from dizoo.distar.envs.distar_env import DIStarEnv
from ding.framework.middleware.tests.mock_for_test import DIStarMockPolicy, DIStarMockPolicyCollect, \
    battle_inferencer_for_distar, battle_rolloutor_for_distar
from distar.ctools.utils import read_config
from unittest.mock import patch

N_ACTORS = 1
N_LEARNERS = 2

cfg = deepcopy(cfg)
env_cfg = read_config('./test_distar_config.yaml')


class PrepareTest():

    @classmethod
    def get_env_fn(cls):
        return DIStarEnv(env_cfg)

    @classmethod
    def get_env_supervisor(cls):
        env = EnvSupervisor(
            type_=ChildType.THREAD,
            env_fn=[cls.get_env_fn for _ in range(cfg.env.collector_env_num)],
            **cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    @classmethod
    def policy_fn(cls):
        model = VAC(**cfg.policy.model)
        policy = DIStarMockPolicy(cfg.policy, model=model)
        return policy

    @classmethod
    def collect_policy_fn(cls):
        policy = DIStarMockPolicyCollect()
        return policy


def _main():
    league = MockLeague(cfg.policy.other.league)

    with task.start(async_mode=True, ctx=BattleContext()):
        with patch("ding.framework.middleware.collector.battle_inferencer", battle_inferencer_for_distar):
            with patch("ding.framework.middleware.collector.battle_rolloutor", battle_rolloutor_for_distar):
                print("node id:", task.router.node_id)
                if task.router.node_id == 0:
                    task.use(LeagueCoordinator(league))
                elif task.router.node_id <= N_ACTORS:
                    task.use(StepLeagueActor(cfg, PrepareTest.get_env_supervisor, PrepareTest.collect_policy_fn))
                else:
                    n_players = len(league.active_players_ids)
                    player = league.active_players[task.router.node_id % n_players]
                    learner = LeagueLearner(cfg, PrepareTest.policy_fn, player)
                    learner._learner._tb_logger = MockLogger()
                    task.use(learner)

                task.run()


@pytest.mark.unittest
def test_league_actor():
    Parallel.runner(n_parallel_workers=N_ACTORS + N_LEARNERS + 1, protocol="tcp", topology="mesh")(_main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=N_ACTORS + N_LEARNERS + 1, protocol="tcp", topology="mesh")(_main)

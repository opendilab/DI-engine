from copy import deepcopy
from time import sleep
import pytest
from ding.envs import BaseEnvManager
from ding.framework.context import BattleContext
from ding.framework.middleware import StepLeagueActor, LeagueCoordinator, LeagueLearner

from ding.model import VAC
from ding.framework.task import task, Parallel
from ding.framework.middleware.tests import cfg, MockLeague, MockLogger
from dizoo.distar.envs.distar_env import DIStarEnv
from ding.framework.middleware.tests.mock_for_test import DIStarMockPolicy, DIStarMockPolicyCollect, \
    battle_inferencer_for_distar, battle_rolloutor_for_distar
from distar.ctools.utils import read_config
from unittest.mock import patch

N_ACTORS = 2
N_LEARNERS = 2


def prepare_test():
    global cfg
    cfg = deepcopy(cfg)
    env_cfg = read_config('./test_distar_config.yaml')

    def env_fn():
        # subprocess env manager
        env = BaseEnvManager(
            env_fn=[lambda: DIStarEnv(env_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    def policy_fn():
        model = VAC(**cfg.policy.model)
        policy = DIStarMockPolicy(cfg.policy, model=model)
        return policy

    def collect_policy_fn():
        policy = DIStarMockPolicyCollect()
        return policy

    return cfg, env_fn, policy_fn, collect_policy_fn


def _main():
    cfg, env_fn, policy_fn, collect_policy_fn = prepare_test()
    league = MockLeague(cfg.policy.other.league)

    with task.start(async_mode=True, ctx=BattleContext()):
        with patch("ding.framework.middleware.collector.battle_inferencer", battle_inferencer_for_distar):
            with patch("ding.framework.middleware.collector.battle_rolloutor", battle_rolloutor_for_distar):
                print("node id:", task.router.node_id)
                if task.router.node_id == 0:
                    task.use(LeagueCoordinator(league))
                elif task.router.node_id <= N_ACTORS:
                    task.use(StepLeagueActor(cfg, env_fn, collect_policy_fn))
                else:
                    n_players = len(league.active_players_ids)
                    player = league.active_players[task.router.node_id % n_players]
                    learner = LeagueLearner(cfg, policy_fn, player)
                    learner._learner._tb_logger = MockLogger()
                    task.use(learner)

                task.run()


@pytest.mark.unittest
def test_league_actor():
    Parallel.runner(n_parallel_workers=N_ACTORS + N_LEARNERS + 1, protocol="tcp", topology="mesh")(_main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=N_ACTORS + N_LEARNERS + 1, protocol="tcp", topology="mesh")(_main)

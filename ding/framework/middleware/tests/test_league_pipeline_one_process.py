from copy import deepcopy
from time import sleep
import pytest
from copy import deepcopy
from ding.envs import BaseEnvManager
from ding.framework.context import BattleContext
from ding.framework.middleware.tests.league_config import cfg
from ding.framework.middleware import LeagueActor, StepLeagueActor, LeagueCoordinator

from ding.envs import BaseEnvManager
from ding.model import VAC
from ding.framework.task import task, Parallel
from ding.framework.middleware import LeagueCoordinator, LeagueActor, LeagueLearner
from ding.framework.middleware.tests import cfg, MockLeague, MockLogger
from dizoo.distar.envs.distar_env import DIStarEnv
from ding.framework.middleware.tests.mock_for_test import DIStarMockPolicy, DIStarMockPolicyCollect, battle_inferencer_for_distar, battle_rolloutor_for_distar
from distar.ctools.utils import read_config
from unittest.mock import patch
import os

def prepare_test():
    global cfg
    cfg = deepcopy(cfg)
    env_cfg = read_config('./test_distar_config.yaml')

    def env_fn():
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
    n_players = len(league.active_players_ids)
    print(n_players)

    with task.start(async_mode=True, ctx=BattleContext()):
        with patch("ding.framework.middleware.collector.battle_inferencer", battle_inferencer_for_distar):
            with patch("ding.framework.middleware.collector.battle_rolloutor", battle_rolloutor_for_distar):
                player_0 = league.active_players[0]
                learner_0 = LeagueLearner(cfg, policy_fn, player_0)
                learner_0._learner._tb_logger = MockLogger()

                player_1 = league.active_players[1]
                learner_1 = LeagueLearner(cfg, policy_fn, player_1)
                learner_1._learner._tb_logger = MockLogger()

                task.use(LeagueCoordinator(league))
                task.use(StepLeagueActor(cfg, env_fn, collect_policy_fn))
                task.use(learner_0)
                task.use(learner_1)

                task.run(max_step=300)


@pytest.mark.unittest
def test_league_actor():
    _main()


if __name__ == '__main__':
    _main()

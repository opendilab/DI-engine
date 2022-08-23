import logging
import pytest
from easydict import EasyDict
from copy import deepcopy
from ding.data import DequeBuffer
from ding.envs import BaseEnvManager, EnvSupervisor
from ding.framework.supervisor import ChildType
from ding.framework.context import BattleContext
from ding.framework.middleware import StepLeagueActor, LeagueCoordinator, LeagueLearnerCommunicator, data_pusher, OffPolicyLearner
from ding.framework.middleware.tests.mock_for_test import DIStarMockPolicy, DIStarMockPolicyCollect
from ding.framework.middleware.functional.collector import battle_inferencer_for_distar, battle_rolloutor_for_distar
from ding.framework.task import task, Parallel
from ding.league.v2 import BaseLeague
from dizoo.distar.config import distar_cfg
from dizoo.distar.envs.distar_env import DIStarEnv
from unittest.mock import patch
from dizoo.distar.policy.distar_policy import DIStarPolicy
from ding.data.buffer.middleware import use_time_check

env_cfg = dict(
    actor=dict(job_type='train', ),
    env=dict(
        map_name='KingsCove',
        player_ids=['agent1', 'agent2'],
        races=['zerg', 'zerg'],
        map_size_resolutions=[True, True],  # if True, ignore minimap_resolutions
        minimap_resolutions=[[160, 152], [160, 152]],
        realtime=False,
        replay_dir='.',
        random_seed='none',
        game_steps_per_episode=100000,
        update_bot_obs=False,
        save_replay_episodes=1,
        update_both_obs=False,
        version='4.10.0',
    ),
)
env_cfg = EasyDict(env_cfg)
cfg = deepcopy(distar_cfg)


class PrepareTest():

    @classmethod
    def get_env_fn(cls):
        return DIStarEnv(env_cfg)

    @classmethod
    def get_env_supervisor(cls):
        for _ in range(10):
            try:
                env = EnvSupervisor(
                    type_=ChildType.THREAD,
                    env_fn=[cls.get_env_fn for _ in range(cfg.env.collector_env_num)],
                    **cfg.env.manager
                )
                env.seed(cfg.seed)
                return env
            except Exception as e:
                print(e)
                continue

    @classmethod
    def policy_fn(cls):
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['learn'])
        return policy

    @classmethod
    def collect_policy_fn(cls):
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['collect'])
        return policy


def coordinator():
    coordinator_league = BaseLeague(cfg.policy.other.league)
    task.use(LeagueCoordinator(cfg, coordinator_league))


def learner():
    league = BaseLeague(cfg.policy.other.league)
    N_PLAYERS = len(league.active_players_ids)

    cfg.policy.collect.unroll_len = 1
    player = league.active_players[task.router.node_id % N_PLAYERS]
    del league

    buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
    buffer_.use(use_time_check(buffer_, max_use=cfg.policy.other.replay_buffer.max_use))
    policy = PrepareTest.policy_fn()

    task.use(LeagueLearnerCommunicator(cfg, policy.learn_mode, player))
    task.use(data_pusher(cfg, buffer_))
    task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))


def actor():
    task.use(StepLeagueActor(cfg, PrepareTest.get_env_supervisor, PrepareTest.collect_policy_fn))


def main():
    logging.getLogger().setLevel(logging.INFO)
    league = BaseLeague(cfg.policy.other.league)
    N_PLAYERS = len(league.active_players_ids)
    del league
    print("League: n_players =", N_PLAYERS)

    with task.start(async_mode=False, ctx=BattleContext()),\
      patch("ding.framework.middleware.collector.battle_inferencer", battle_inferencer_for_distar),\
      patch("ding.framework.middleware.collector.battle_rolloutor", battle_rolloutor_for_distar):
        print("node id:", task.router.node_id)
        if task.router.node_id == 0:
            coordinator()
        elif task.router.node_id <= N_PLAYERS:
            learner()
        else:
            actor()
        task.run()


@pytest.mark.unittest
def test_league_pipeline():
    Parallel.runner(n_parallel_workers=7, protocol="tcp", topology="mesh")(main)


if __name__ == "__main__":
    Parallel.runner(n_parallel_workers=7, protocol="tcp", topology="mesh")(main)

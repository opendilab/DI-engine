from copy import deepcopy
from dataclasses import dataclass
from time import sleep
import time
import pytest
import logging
from typing import Any

from ding.data import DequeBuffer
from ding.envs import BaseEnvManager
from ding.framework.context import BattleContext
from ding.framework import EventEnum
from ding.framework.task import task, Parallel
from ding.framework.middleware import data_pusher, OffPolicyLearner, LeagueLearnerCommunicator
from ding.framework.middleware.functional.actor_data import *
from ding.framework.middleware.tests.mock_for_test import DIStarMockPolicy
from ding.league.v2 import BaseLeague
from ding.utils import log_every_sec
from dizoo.distar.config import distar_cfg
from dizoo.distar.envs import fake_rl_traj_with_last
from dizoo.distar.envs.distar_env import DIStarEnv
from distar.ctools.utils import read_config


def prepare_test():
    global distar_cfg
    cfg = deepcopy(distar_cfg)
    env_cfg = read_config('./test_distar_config.yaml')

    def env_fn():
        # subprocess env manager
        env = BaseEnvManager(
            env_fn=[lambda: DIStarEnv(env_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    def policy_fn():
        policy = DIStarMockPolicy(DIStarMockPolicy.default_config(), enable_field=['learn'])
        return policy

    return cfg, env_fn, policy_fn


def coordinator_mocker():
    task.on(EventEnum.LEARNER_SEND_META, lambda x: logging.info("test: {}".format(x)))
    task.on(EventEnum.LEARNER_SEND_MODEL, lambda x: logging.info("test: send model success"))

    def _coordinator_mocker(ctx):
        sleep(10)

    return _coordinator_mocker


@dataclass
class TestActorData:
    env_step: int
    train_data: Any


def actor_mocker(league):

    def _actor_mocker(ctx):
        n_players = len(league.active_players_ids)
        player = league.active_players[(task.router.node_id + 2) % n_players]
        log_every_sec(logging.INFO, 5, "Actor: actor player: {}".format(player.player_id))
        for _ in range(24):
            meta = ActorDataMeta(player_total_env_step=0, actor_id=0, send_wall_time=time.time())
            data = fake_rl_traj_with_last()
            actor_data = ActorData(meta=meta, train_data=[ActorEnvTrajectories(env_id=0, trajectories=[data])])
            task.emit(EventEnum.ACTOR_SEND_DATA.format(player=player.player_id), actor_data)
        sleep(9)

    return _actor_mocker


def _main():
    logging.getLogger().setLevel(logging.INFO)
    cfg, env_fn, policy_fn = prepare_test()
    league = BaseLeague(cfg.policy.other.league)
    n_players = len(league.active_players_ids)
    print("League: n_players: ", n_players)

    with task.start(async_mode=True, ctx=BattleContext()):
        if task.router.node_id == 0:
            task.use(coordinator_mocker())
        elif task.router.node_id <= 1:
            task.use(actor_mocker(league))
        else:
            cfg.policy.collect.unroll_len = 1
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            player = league.active_players[task.router.node_id % n_players]
            policy = policy_fn()
            task.use(LeagueLearnerCommunicator(cfg, policy.learn_mode, player))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.run(max_step=30)


@pytest.mark.unittest
def test_league_learner():
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="mesh")(_main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="mesh")(_main)

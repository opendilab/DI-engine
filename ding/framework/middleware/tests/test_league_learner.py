from copy import deepcopy
from time import sleep
import torch
import pytest
import logging

from ding.envs import BaseEnvManager
from ding.data import DequeBuffer
from ding.framework.context import BattleContext
from ding.framework import EventEnum
from ding.framework.task import task, Parallel
from ding.framework.middleware import OffPolicyLeagueLearner, data_pusher,\
    OffPolicyLearner, LeagueLearnerExchanger
from ding.framework.middleware.functional.actor_data import ActorData
from ding.framework.middleware.tests import cfg, MockLeague, MockLogger
from ding.framework.middleware.tests.mock_for_test import DIStarMockPolicy
from dizoo.distar.envs.distar_env import DIStarEnv
from distar.ctools.utils import read_config
from dizoo.distar.envs import fake_rl_data_batch_with_last

logging.getLogger().setLevel(logging.INFO)


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
        policy = DIStarMockPolicy(DIStarMockPolicy.default_config(), enable_field=['learn'])
        return policy

    return cfg, env_fn, policy_fn


def coordinator_mocker():
    task.on(EventEnum.LEARNER_SEND_META, lambda x: print("test:", x))
    task.on(EventEnum.LEARNER_SEND_MODEL, lambda x: print("test: send model success"))

    def _coordinator_mocker(ctx):
        sleep(10)

    return _coordinator_mocker


def actor_mocker(league):

    def _actor_mocker(ctx):
        n_players = len(league.active_players_ids)
        player = league.active_players[(task.router.node_id + 2) % n_players]
        print("actor player:", player.player_id)
        for _ in range(3):
            data = fake_rl_data_batch_with_last()
            actor_data = ActorData(env_step=0, train_data=data)
            task.emit(EventEnum.ACTOR_SEND_DATA.format(player=player.player_id), actor_data)
        sleep(9)

    return _actor_mocker


def _main():
    cfg, env_fn, policy_fn = prepare_test()
    league = MockLeague(cfg.policy.other.league)

    with task.start(async_mode=True, ctx=BattleContext()):
        if task.router.node_id == 0:
            task.use(coordinator_mocker())
        elif task.router.node_id <= 2:
            task.use(actor_mocker(league))
        else:
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            n_players = len(league.active_players_ids)
            print("League: n_players: ", n_players)
            player = league.active_players[task.router.node_id % n_players]
            policy = policy_fn()
            task.use(LeagueLearnerExchanger(cfg, policy.learn_mode, player))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.run(max_step=200)


@pytest.mark.unittest
def test_league_learner():
    Parallel.runner(n_parallel_workers=5, protocol="tcp", topology="mesh")(_main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=5, protocol="tcp", topology="mesh")(_main)

from copy import deepcopy
from dataclasses import dataclass
from time import sleep
import time
import pytest
import logging
from typing import Any
from unittest.mock import patch
from typing import Callable, Optional

from ding.framework.context import BattleContext
from ding.framework import EventEnum
from ding.framework.task import task, Parallel
from ding.framework.middleware import LeagueLearnerCommunicator, LearnerModel
from ding.framework.middleware.functional.actor_data import *
from ding.framework.middleware.tests.mock_for_test import league_cfg

from ding.model import VAC
from ding.policy.ppo import PPOPolicy

PLAYER_ID = "test_player"

def prepare_test():
    global league_cfg
    cfg = deepcopy(league_cfg)

    def policy_fn():
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        return policy

    return cfg, policy_fn


@dataclass
class TestActorData:
    env_step: int
    train_data: Any


class MockFileStorage:

    def __init__(self, path: str) -> None:
        self.path = path
    
    def save(self, data: Any) -> bool:
        assert isinstance(data, dict)


class MockPlayer:

    def __init__(self) -> None:
        self.player_id = PLAYER_ID
        self.total_agent_step = 0

    def is_trained_enough(self) -> bool:
        return True


def coordinator_mocker():

    test_cases = {
        "on_learner_meta": False
    }

    def on_learner_meta(player_meta):
        assert player_meta.player_id == PLAYER_ID
        test_cases["on_learner_meta"] = True


    task.on(EventEnum.LEARNER_SEND_META.format(player=PLAYER_ID), on_learner_meta)

    def _coordinator_mocker(ctx):
        sleep(0.8)
        assert all(test_cases.values())

    return _coordinator_mocker


def actor_mocker():

    test_cases = {
        "on_learner_model": False
    }

    def on_learner_model(learner_model):
        assert isinstance(learner_model, LearnerModel)
        assert learner_model.player_id == PLAYER_ID
        test_cases["on_learner_model"] = True
    
    task.on(EventEnum.LEARNER_SEND_MODEL.format(player=PLAYER_ID), on_learner_model)

    def _actor_mocker(ctx):
        sleep(0.2)
        player = MockPlayer()
        for _ in range(10):
            meta = ActorDataMeta(player_total_env_step=0, actor_id=0, send_wall_time=time.time())
            data = []
            actor_data = ActorData(meta=meta, train_data=[ActorEnvTrajectories(env_id=0, trajectories=[data])])
            task.emit(EventEnum.ACTOR_SEND_DATA.format(player=player.player_id), actor_data)
        
        sleep(0.8)
        assert all(test_cases.values())

    return _actor_mocker


def _main():
    logging.getLogger().setLevel(logging.INFO)
    cfg, policy_fn = prepare_test()

    with task.start(async_mode=False, ctx=BattleContext()):
        if task.router.node_id == 0:
            task.use(coordinator_mocker())
        elif task.router.node_id <= 1:
            task.use(actor_mocker())
        else:
            player = MockPlayer()
            policy = policy_fn()
            with patch("ding.framework.storage.FileStorage", MockFileStorage):
                learner_communicator = LeagueLearnerCommunicator(cfg, policy.learn_mode, player)
                sleep(0.5)
                assert len(learner_communicator._cache) == 10
                task.use(learner_communicator)
                sleep(0.1)
        task.run(max_step=1)


@pytest.mark.unittest
def test_league_learner_communicator():
    Parallel.runner(n_parallel_workers=3, protocol="tcp", topology="mesh")(_main)

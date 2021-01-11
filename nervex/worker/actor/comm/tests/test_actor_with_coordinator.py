import pytest
import os
import time
from threading import Thread
from multiprocessing import Process
import torch

from nervex.worker import Coordinator, create_comm_actor
from nervex.worker.learner.comm import NaiveLearner
from nervex.utils import read_config, lists_to_dicts
from nervex.interaction.slave import Slave, TaskFail
from nervex.config import parallel_local_default_config, parallel_transform

DATA_PREFIX = 'SLAVE_ACTOR_DATA'


@pytest.fixture(scope='function')
def setup_config():
    cfg = parallel_local_default_config
    cfg = parallel_transform(cfg)
    return cfg


@pytest.fixture(scope='function')
def setup_actor(setup_config):
    actor = {}
    for k, v in setup_config.items():
        if 'actor' in k:
            actor[k] = create_comm_actor(v)
            actor[k].start()
    yield actor
    for a in actor.values():
        a.close()


@pytest.fixture(scope='function')
def setup_learner(setup_config):
    cfg = setup_config.coordinator.interaction.learner
    learner = {}
    for _, (name, host, port) in cfg.items():
        learner[name] = NaiveLearner(host, port)
        learner[name].start()
    yield learner
    for l in learner.values():
        l.close()


@pytest.mark.unittest
class TestActorWithCoordinator:

    def test_naive(self, setup_config, setup_actor, setup_learner):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        os.popen('rm -rf env_*_*')
        os.popen('rm -rf test.pth')
        assert len(setup_actor) == len(setup_config.coordinator.interaction.actor)
        try:
            coordinator = Coordinator(setup_config.coordinator)
            coordinator.start()
            while True:
                if setup_actor['actor0']._actor is not None:
                    break
                time.sleep(0.5)
            torch.save(
                {
                    'model': setup_actor['actor0']._actor.policy.state_dict_handle()['model'].state_dict(),
                    'iter': 0
                }, 'test.pth'
            )
            while True:
                commander = coordinator._commander
                if commander._learner_task_finish_count == 1 and commander._actor_task_finish_count == 2:
                    break
                time.sleep(0.5)
            coordinator.close()
        except Exception as e:
            os.popen('rm -rf {}*'.format(DATA_PREFIX))
            assert False, e

        assert len(coordinator._replay_buffer) == 0
        learner_task_ids = [i for i in coordinator._historical_task if 'learner' in i]
        for i in learner_task_ids:
            assert len(coordinator._commander._learner_info[i]) == 5
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        os.popen('rm -rf env_*_*')

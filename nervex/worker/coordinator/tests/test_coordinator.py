import pytest
import os
import time
from threading import Thread
from multiprocessing import Process
from nervex.worker import Coordinator
from nervex.worker.learner.comm import NaiveLearner
from nervex.worker.actor.comm import NaiveActor
from nervex.utils import lists_to_dicts
from nervex.interaction.slave import Slave, TaskFail
from nervex.config import coordinator_default_config

DATA_PREFIX = 'SLAVE_ACTOR_DATA'


@pytest.fixture(scope='function')
def setup_config():
    cfg = coordinator_default_config
    cfg.interaction.port += 100
    for k in cfg.interaction.actor:
        cfg.interaction.actor[k][2] += 100
    for k in cfg.interaction.learner:
        cfg.interaction.learner[k][2] += 100
    return cfg


@pytest.fixture(scope='function')
def setup_actor(setup_config):
    cfg = setup_config.interaction.actor
    actor = {}
    for _, (name, host, port) in cfg.items():
        actor[name] = NaiveActor(host, port)
        actor[name].start()
    yield actor
    for a in actor.values():
        a.close()


@pytest.fixture(scope='function')
def setup_learner(setup_config):
    cfg = setup_config.interaction.learner
    learner = {}
    for _, (name, host, port) in cfg.items():
        learner[name] = NaiveLearner(host, port)
        learner[name].start()
    yield learner
    for l in learner.values():
        l.close()


@pytest.mark.unittest
class TestCoordinator:

    def test_naive(self, setup_config, setup_actor, setup_learner):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        assert len(setup_actor) == len(setup_config.interaction.actor)
        assert len(setup_learner) == len(setup_config.interaction.learner)
        try:
            coordinator = Coordinator(setup_config)
            coordinator.start()
            while True:
                if coordinator._commander._learner_task_finish_count == 1:
                    break
                time.sleep(0.5)
            coordinator.close()
        except Exception as e:
            os.popen('rm -rf {}*'.format(DATA_PREFIX))
            assert False, e

        actor_task_ids = [t for t in coordinator._historical_task if 'actor' in t]
        for i in range(1, 21):
            for t in actor_task_ids:
                assert os.path.exists('{}_{}_{}'.format(DATA_PREFIX, t, i))
        assert os.path.exists('{}_final_model.pth'.format(DATA_PREFIX))
        assert len(coordinator._replay_buffer) == 0
        learner_task_ids = [i for i in coordinator._historical_task if 'learner' in i]
        for i in learner_task_ids:
            assert len(coordinator._commander._learner_info[i]) == 5
        os.popen('rm -rf {}*'.format(DATA_PREFIX))

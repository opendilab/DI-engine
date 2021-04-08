import pytest
import os
import time
from multiprocessing import Process

from nervex.worker import Coordinator, create_comm_learner
from nervex.worker.collector.comm import NaiveCollector
from nervex.utils import lists_to_dicts
from nervex.config import parallel_local_default_config, parallel_transform

DATA_PREFIX = 'SLAVE_COLLECTOR_DATA_LEARNER_TEST'


@pytest.fixture(scope='function')
def setup_config():
    return parallel_transform(parallel_local_default_config)


@pytest.fixture(scope='function')
def setup_collector(setup_config):
    cfg = setup_config.coordinator.interaction.collector
    collector = {}
    for _, (name, host, port) in cfg.items():
        collector[name] = NaiveCollector(host, port, prefix=DATA_PREFIX)
        collector[name].start()
    yield collector
    for a in collector.values():
        a.close()


@pytest.fixture(scope='function')
def setup_learner(setup_config):
    learner = {}
    for k, v in setup_config.items():
        if 'learner' in k:
            learner[k] = create_comm_learner(v)
            learner[k].start()
    yield learner
    for l in learner.values():
        l.close()


@pytest.mark.unittest(rerun=5)
class TestLearnerWithCoordinator:

    def test_naive(self, setup_config, setup_collector, setup_learner):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        assert len(setup_collector) == len(setup_config.coordinator.interaction.collector)
        try:
            coordinator = Coordinator(setup_config.coordinator)
            coordinator.start()
            while True:
                if coordinator._commander._learner_task_finish_count == 1:
                    break
                time.sleep(0.5)
            coordinator.close()
        except Exception as e:
            os.popen('rm -rf {}*'.format(DATA_PREFIX))
            assert False, e

        collector_task_ids = [t for t in coordinator._historical_task if 'collector' in t]
        for i in range(1, 21):
            for t in collector_task_ids:
                assert os.path.exists('{}_{}_{}'.format(DATA_PREFIX, t, i))
        assert len(coordinator._replay_buffer) == 0
        learner_task_ids = [i for i in coordinator._historical_task if 'learner' in i]
        for i in learner_task_ids:
            assert len(coordinator._commander._learner_info[i]) == setup_config.coordinator.commander.max_iterations
        os.popen('rm -rf {}*'.format(DATA_PREFIX))

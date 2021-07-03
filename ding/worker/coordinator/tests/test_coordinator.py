import pytest
import os
import time
from ding.worker import Coordinator
from ding.worker.learner.comm import NaiveLearner
from ding.worker.collector.comm import NaiveCollector
from ding.utils import find_free_port
from ding.config import compile_config_parallel
from ding.config.utils import parallel_test_main_config, parallel_test_create_config, parallel_test_system_config

DATA_PREFIX = 'SLAVE_COLLECTOR_DATA_COORDINATOR_TEST'


@pytest.fixture(scope='function')
def setup_config():
    return compile_config_parallel(
        parallel_test_main_config, create_cfg=parallel_test_create_config, system_cfg=parallel_test_system_config
    )


@pytest.fixture(scope='function')
def setup_collector(setup_config):
    cfg = setup_config.system.coordinator.collector
    collector = {}
    for _, (name, host, port) in cfg.items():
        collector[name] = NaiveCollector(host, port, prefix=DATA_PREFIX)
        collector[name].start()
    yield collector
    for a in collector.values():
        a.close()


@pytest.fixture(scope='function')
def setup_learner(setup_config):
    cfg = setup_config.system.coordinator.learner
    learner = {}
    for _, (name, host, port) in cfg.items():
        learner[name] = NaiveLearner(host, port, prefix=DATA_PREFIX)
        learner[name].start()
    yield learner
    for l in learner.values():
        l.close()


@pytest.mark.unittest(rerun=5)
class TestCoordinator:

    def test_naive(self, setup_config, setup_collector, setup_learner):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        assert len(setup_collector) == len(setup_config.system.coordinator.collector)
        assert len(setup_learner) == len(setup_config.system.coordinator.learner)
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

        collector_task_ids = [t for t in coordinator._historical_task if 'collector' in t]
        for i in range(1, 21):
            for t in collector_task_ids:
                assert os.path.exists('{}_{}_{}'.format(DATA_PREFIX, t, i))
        assert os.path.exists('{}_final_model.pth'.format(DATA_PREFIX))
        assert len(coordinator._replay_buffer) == 0
        learner_task_ids = [i for i in coordinator._historical_task if 'learner' in i]
        for i in learner_task_ids:
            assert len(coordinator._commander._learner_info[i]) == 5
        os.popen('rm -rf {}*'.format(DATA_PREFIX))

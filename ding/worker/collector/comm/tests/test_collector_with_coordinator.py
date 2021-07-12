import pytest
import os
import time
from threading import Thread
from multiprocessing import Process
import torch

from ding.worker import Coordinator, create_comm_collector
from ding.worker.learner.comm import NaiveLearner
from ding.utils import lists_to_dicts
from ding.interaction.slave import Slave, TaskFail
from ding.config import compile_config_parallel
from ding.config.utils import parallel_test_main_config, parallel_test_create_config, parallel_test_system_config

DATA_PREFIX = 'SLAVE_COLLECTOR_DATA_COLLECTOR_TEST'


@pytest.fixture(scope='function')
def setup_config():
    return compile_config_parallel(
        parallel_test_main_config, create_cfg=parallel_test_create_config, system_cfg=parallel_test_system_config
    )


@pytest.fixture(scope='function')
def setup_collector(setup_config):
    collector = {}
    for k, v in setup_config.system.items():
        if 'collector' in k:
            collector[k] = create_comm_collector(v)
            collector[k].start()
    yield collector
    time.sleep(1)  # avoid collector is not closed but comm collector receive close signal
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
    time.sleep(1)
    for l in learner.values():
        l.close()


@pytest.mark.unittest
class TestCollectorWithCoordinator:

    def test_naive(self, setup_config, setup_collector, setup_learner):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        os.popen('rm -rf env_*_*')
        os.popen('rm -rf test.pth')
        assert len(setup_collector) == len(setup_config.system.coordinator.collector)
        try:
            coordinator = Coordinator(setup_config)
            coordinator.start()
            while True:
                if setup_collector['collector0']._collector is not None:
                    break
                time.sleep(0.5)
            torch.save(
                {
                    'model': setup_collector['collector0']._collector.policy.state_dict()['model'],
                    'iter': 0
                }, 'test.pth'
            )
            while True:
                commander = coordinator._commander
                if commander._learner_task_finish_count >= 1 and commander._collector_task_finish_count >= 2:
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

import pytest
import os
import copy
import time
from threading import Thread
import json
from queue import Queue
from flask import Flask, request

from ding.worker import Coordinator
from ding.worker.learner.comm import NaiveLearner
from ding.worker.collector.comm import NaiveCollector
from ding.utils import find_free_port
from ding.config import compile_config_parallel
from ding.config.utils import parallel_test_main_config, parallel_test_create_config, parallel_test_system_config

DATA_PREFIX = 'SLAVE_COLLECTOR_DATA_FAKE_OPERATOR_TEST'
init_replicas_request = {
    "collectors": {
        "cpu": "0.5",
        "memory": "200Mi",
        "replicas": 2,
    },
    "learners": {
        "cpu": "0.5",
        "memory": "200Mi",
        "gpu": "0",
        "replicas": 1,
    },
}
api_version = 'v1alpha1'
system_addr = 'https://0.0.0.0:14502'


def create_app(creator):
    app = Flask(__name__)

    @app.route('/{}/replicas'.format(api_version), methods=['POST'])
    def post_replicas():
        data = json.loads(request.data.decode())
        collectors = data['collectors']["replicas"]
        learners = data['learners']["replicas"]
        creator.set_target_source(learners, collectors)
        return {'success': True, 'code': 0, 'message': '', 'data': ''}

    @app.route('/{}/replicas'.format(api_version), methods=['GET'])
    def get_replicas():
        data = json.loads(request.data.decode())
        return {'success': True, 'code': 0, 'message': '', 'data': creator.current_resource}

    return app


@pytest.fixture(scope='function')
def setup_config():
    cfg = compile_config_parallel(
        parallel_test_main_config, create_cfg=parallel_test_create_config, system_cfg=parallel_test_system_config
    )
    cfg.system.coordinator.operator_server = dict(
        system_addr=system_addr,
        api_version=api_version,
        init_replicas_request=init_replicas_request,
        collector_target_num=len(cfg.system.coordinator.collector),
        learner_target_num=len(cfg.system.coordinator.learner),
    )
    return cfg


class Creator:

    def __init__(self, learner_addr, collector_addr):
        self.learner_addr = learner_addr
        self.collector_addr = collector_addr
        self.collector_demand = Queue()
        self.learner_demand = Queue()
        self.learners = {}
        self.collectors = {}
        self.end_flag = False

    def set_target_source(self, learner_target, collector_target):
        print('set_target_source', learner_target, collector_target)
        time.sleep(3)  # simulate
        self.collector_demand.put(collector_target)
        self.learner_demand.put(learner_target)

    def start(self):
        while not self.end_flag:
            if self.learner_demand.empty() and self.collector_demand.empty():
                time.sleep(0.1)
                continue
            else:
                learner_demand, collector_demand = None, None
                if not self.learner_demand.empty():
                    learner_demand = self.learner_demand.get()
                if not self.collector_demand.empty():
                    collector_demand = self.collector_demand.get()

                for i in range(collector_demand):
                    name, host, port = self.collector_addr[i]
                    self.collectors[name] = NaiveCollector(host, port, prefix=DATA_PREFIX)
                    self.collectors[name].start()
                for i in range(learner_demand):
                    name, host, port = self.learner_addr[i]
                    self.learners[name] = NaiveLearner(host, port, prefix=DATA_PREFIX)
                    self.learners[name].start()

    def close(self):
        self.end_flag = True
        time.sleep(1)
        for t in self.learners.values():
            t.close()
        for t in self.collectors.values():
            t.close()

    @property
    def current_resource(self):
        collectors = {k: {} for k in self.collectors}
        learners = {k: {} for k in self.learners}
        return {"collectors": collectors, 'learners': learners}


@pytest.fixture(scope='function')
def setup_operator_server(setup_config):
    host, port = system_addr.split("https://")[1].split(":")
    port = int(port)
    learner_addr = copy.deepcopy(setup_config.system.coordinator.learner)
    learner_addr = list(learner_addr.values())
    for i in range(len(learner_addr)):
        learner_addr[i][0] = '{}:{}'.format(learner_addr[i][1], learner_addr[i][2])
    collector_addr = copy.deepcopy(setup_config.system.coordinator.collector)
    collector_addr = list(collector_addr.values())
    for i in range(len(collector_addr)):
        collector_addr[i][0] = '{}:{}'.format(collector_addr[i][1], collector_addr[i][2])
    print(learner_addr, collector_addr)

    creator = Creator(learner_addr, collector_addr)
    creator_start_thread = Thread(target=creator.start, args=(), daemon=True)
    creator_start_thread.start()

    app = create_app(creator)
    app_run_thread = Thread(target=app.run, args=(host, port), daemon=True)
    app_run_thread.start()
    yield app
    creator.close()
    print('end')


@pytest.mark.unittest
class TestCoordinatorFakeOperator:

    def test_naive(self, setup_config, setup_operator_server):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        # learner/collector is created by operator-server
        setup_config.system.coordinator.learner = {}
        setup_config.system.coordinator.collector = {}

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

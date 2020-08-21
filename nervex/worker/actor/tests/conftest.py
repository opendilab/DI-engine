import pytest
import uuid
import torch
import os
import yaml
from easydict import EasyDict
from threading import Thread
from nervex.system import Manager, create_manager_app, Coordinator, create_coordinator_app
from nervex.system.coordinator import JobState


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../sumowj3_actor_default_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


@pytest.fixture(scope='function')
def setup_manager(setup_config):
    manager = Manager(setup_config)
    manager_app = create_manager_app(manager)

    def run():
        try:
            manager_app.run(host=setup_config.system.manager_ip, port=setup_config.system.manager_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run, args=())
    launch_thread.daemon = True
    launch_thread.start()
    yield launch_thread
    # clean manager source


class FakeCoordinator(Coordinator):
    def deal_with_ask_for_job(self, manager_uid, actor_uid):
        fake_job = {
            'job_id': str(uuid.uuid1()),
            'episode_num': 2,
            'env_num': 3,
            'agent_num': 1,
            'agent': {
                '0': {
                    'name': '0',
                    'model': {},
                    'agent_update_path': 'model_placeholder'
                }
            },
            'learner_uid': 'learner_uid_placeholder',
            'agent_update_freq': 30,
            'data_push_length': 16,
            'compressor': 'none',
        }
        self.job_queue.put(fake_job)
        return super().deal_with_ask_for_job(manager_uid, actor_uid)

    def deal_with_finish_job(self, manager_uid, actor_uid, job_id, result):
        assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        self.job_record[job_id]['state'] = JobState.finish
        self._logger.info('job({}) finish with result: {}'.format(job_id, result))
        return True

    def deal_with_get_metadata(self, job_id, metadata):
        return True


@pytest.fixture(scope='function')
def setup_coordinator(setup_config):
    coordinator = FakeCoordinator(setup_config)
    coordinator_app = create_coordinator_app(coordinator)

    def run():
        try:
            coordinator_app.run(host=setup_config.system.coordinator_ip, port=setup_config.system.coordinator_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run, args=())
    launch_thread.daemon = True
    launch_thread.start()
    yield launch_thread
    # clean coordinator source

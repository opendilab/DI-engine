import os
import uuid
from threading import Thread

import pytest
import yaml
from easydict import EasyDict

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
            manager_app.run(
                host=setup_config.system.manager_ip,
                port=setup_config.system.manager_port,
                debug=True,
                use_reloader=False
            )
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
            'env_kwargs': {
                'episode_num': 2,
                'env_num': 4,
                'env_cfg': {
                    'env_type': 'fake'
                },
            },
            'agent_num': 1,
            'agent': {
                '0': {
                    'name': '0',
                    'model': {},
                    'agent_update_path': os.path.join(os.path.dirname(__file__), 'model_placeholder'),
                }
            },
            'learner_uid': 'learner_uid_placeholder',
            'agent_update_freq': 30,
            'compressor': 'none',
            'player_id': ['test'],
            'launch_player': ['test'],
            'forward_kwargs': {
                'eps': 0.9,
            },
            'adder_kwargs': {
                'use_gae': False,
                'data_push_length': 16,
            }
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
            coordinator_app.run(
                host=setup_config.system.coordinator_ip,
                port=setup_config.system.coordinator_port,
                debug=True,
                use_reloader=False
            )
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run, args=())
    launch_thread.daemon = True
    launch_thread.start()
    yield coordinator
    # clean coordinator source
    coordinator.close()

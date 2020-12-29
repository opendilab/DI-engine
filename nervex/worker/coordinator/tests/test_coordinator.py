import pytest
import os
import time
from threading import Thread
from multiprocessing import Process
from nervex.worker import Coordinator
from nervex.utils import read_config
from nervex.interaction.slave import Slave, TaskFail

DATA_PREFIX = 'SLAVE_ACTOR_DATA'


@pytest.fixture(scope='function')
def setup_config():
    return read_config(os.path.join(os.path.dirname(__file__), '../coordinator_default_config.yaml'))


class Actor(Slave):

    def _process_task(self, task):
        task_name = task['name']
        if task_name == 'actor_start_task':
            time.sleep(1)
            self.count = 0
            self.task_info = task['task_info']
            return {'message': 'actor task has started'}
        elif task_name == 'actor_data_task':
            time.sleep(0.1)
            self.count += 1
            data = {'metadata': 'test_field1', 'data_id': 'data_{}'.format(self.count)}
            data['task_id'] = self.task_info['task_id']
            with open('./{}_{}'.format(DATA_PREFIX, self.count), 'w') as f:
                f.writelines(data)
            if self.count == 10:
                data['finish_task'] = True
            else:
                data['finish_task'] = False
            return data
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))


def run_actor(actor):
    actor.start()


@pytest.fixture(scope='function')
def setup_actor(setup_config):
    cfg = setup_config.coordinator.interaction.actor
    actor = {}
    for _, (name, host, port) in cfg.items():
        actor[name] = Actor(host, port)
    return actor


@pytest.mark.unittest
class TestCoordinator:

    def test_naive(self, setup_config, setup_actor):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        for actor_thread in setup_actor.values():
            actor_thread.start()
        coordinator = Coordinator(setup_config)
        coordinator.start()
        time.sleep(2)
        coordinator.close()
        for actor_thread in setup_actor.values():
            actor_thread.close()
        for i in range(1, 11):
            assert os.path.exists('{}_{}'.format(DATA_PREFIX, i))
        os.popen('rm -rf {}*'.format(DATA_PREFIX))

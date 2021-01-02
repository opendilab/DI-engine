import pytest
import os
import time
from threading import Thread
from multiprocessing import Process
import torch

from nervex.worker import Coordinator, create_comm_learner
from nervex.utils import read_config, lists_to_dicts
from nervex.interaction.slave import Slave, TaskFail
from nervex.config import parallel_local_default_config

DATA_PREFIX = 'SLAVE_ACTOR_DATA'


@pytest.fixture(scope='function')
def setup_config():
    return parallel_local_default_config


class Actor(Slave):

    def _process_task(self, task):
        task_name = task['name']
        if task_name == 'resource':
            return {'cpu': '20', 'gpu': '1'}
        elif task_name == 'actor_start_task':
            self.count = 0
            self.task_info = task['task_info']
            return {'message': 'actor task has started'}
        elif task_name == 'actor_data_task':
            self.count += 1
            data_id = './{}_{}_{}'.format(DATA_PREFIX, self.task_info['task_id'], self.count)
            torch.save(self._get_timestep(), data_id)
            data = {'data_id': data_id, 'buffer_id': self.task_info['buffer_id']}
            data['task_id'] = self.task_info['task_id']
            if self.count == 20:
                data['finished_task'] = {'finish': True}
            else:
                data['finished_task'] = None
            return data
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))

    def _get_timestep(self):
        return [
            {
                'obs': torch.rand(4),
                'next_obs': torch.randn(4),
                'reward': torch.randint(0, 2, size=(1, )).float(),
                'action': torch.randint(0, 2, size=(1, )),
                'done': False,
            }
        ]


@pytest.fixture(scope='function')
def setup_actor(setup_config):
    cfg = setup_config.coordinator.interaction.actor
    actor = {}
    for _, (name, host, port) in cfg.items():
        actor[name] = Actor(host, port)
        actor[name].start()
    yield actor
    for a in actor.values():
        a.close()


@pytest.fixture(scope='function')
def setup_learner(setup_config):
    learner = create_comm_learner(setup_config)
    learner.start()
    yield learner
    learner.close()


@pytest.mark.unittest
class TestLearnerWithCoordinator:

    def test_naive(self, setup_config, setup_actor, setup_learner):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        assert len(setup_actor) == len(setup_config.coordinator.interaction.actor)
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

        actor_task_ids = [t for t in coordinator._historical_task if 'actor' in t]
        for i in range(1, 21):
            for t in actor_task_ids:
                assert os.path.exists('{}_{}_{}'.format(DATA_PREFIX, t, i))
        assert len(coordinator._replay_buffer) == 0
        learner_task_ids = [i for i in coordinator._historical_task if 'learner' in i]
        for i in learner_task_ids:
            assert len(coordinator._commander._learner_info[i]) == setup_config.coordinator.max_iterations
        os.popen('rm -rf {}*'.format(DATA_PREFIX))

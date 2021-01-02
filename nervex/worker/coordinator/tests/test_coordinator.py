import pytest
import os
import time
from threading import Thread
from multiprocessing import Process
from nervex.worker import Coordinator
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


class Actor(Slave):

    def _process_task(self, task):
        task_name = task['name']
        if task_name == 'resource':
            return {'cpu': 'xxx', 'gpu': 'xxx'}
        elif task_name == 'actor_start_task':
            time.sleep(1)
            self.count = 0
            self.task_info = task['task_info']
            return {'message': 'actor task has started'}
        elif task_name == 'actor_data_task':
            time.sleep(0.05)
            self.count += 1
            data = {
                'metadata': 'test_field1',
                'data_id': 'data_{}'.format(self.count),
                'buffer_id': self.task_info['buffer_id']
            }
            data['task_id'] = self.task_info['task_id']
            with open('./{}_{}_{}'.format(DATA_PREFIX, self.task_info['task_id'], self.count), 'w') as f:
                f.writelines(data)
            if self.count == 20:
                data['finished_task'] = {'finish': True}
            else:
                data['finished_task'] = None
            return data
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))


class Learner(Slave):

    def _process_task(self, task):
        task_name = task['name']
        if task_name == 'resource':
            return {'cpu': 'xxx', 'gpu': 'xxx'}
        elif task_name == 'learner_start_task':
            time.sleep(1)
            self.task_info = task['task_info']
            self.count = 0
            return {'message': 'learner task has started'}
        elif task_name == 'learner_get_data_task':
            time.sleep(0.01)
            return {'task_id': self.task_info['task_id'], 'buffer_id': self.task_info['buffer_id'], 'batch_size': 2}
        elif task_name == 'learner_learn_task':
            data = task['data']
            if data is None:
                raise TaskFail(result={'message': 'no data'})
            time.sleep(0.1)
            data = lists_to_dicts(data)
            assert 'metadata' in data.keys()
            priority_keys = ['replay_unique_id', 'replay_buffer_idx', 'priority']
            self.count += 1
            ret = {
                'info': {
                    'step': self.count
                },
                'task_id': self.task_info['task_id'],
                'buffer_id': self.task_info['buffer_id']
            }
            ret['info']['priority_info'] = {k: data[k] for k in priority_keys}
            if self.count > 5:
                ret['finished_task'] = {'finish': True, 'buffer_id': self.task_info['buffer_id']}
                os.popen('touch {}_final_model.pth'.format(DATA_PREFIX))
            else:
                ret['finished_task'] = None
            return ret
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))


@pytest.fixture(scope='function')
def setup_actor(setup_config):
    cfg = setup_config.interaction.actor
    actor = {}
    for _, (name, host, port) in cfg.items():
        actor[name] = Actor(host, port)
        actor[name].start()
    yield actor
    for a in actor.values():
        a.close()


@pytest.fixture(scope='function')
def setup_learner(setup_config):
    cfg = setup_config.interaction.learner
    learner = {}
    for _, (name, host, port) in cfg.items():
        learner[name] = Learner(host, port)
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

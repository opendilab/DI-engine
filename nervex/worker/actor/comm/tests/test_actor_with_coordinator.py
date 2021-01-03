import pytest
import os
import time
from threading import Thread
from multiprocessing import Process
import torch

from nervex.worker import Coordinator, create_comm_actor
from nervex.utils import read_config, lists_to_dicts
from nervex.interaction.slave import Slave, TaskFail
from nervex.config import parallel_local_default_config

DATA_PREFIX = 'SLAVE_ACTOR_DATA'


@pytest.fixture(scope='function')
def setup_config():
    cfg = parallel_local_default_config
    cfg.coordinator.interaction.port += 200
    for k in cfg.coordinator.interaction.actor:
        cfg.coordinator.interaction.actor[k][2] += 200
    for k in cfg.coordinator.interaction.learner:
        cfg.coordinator.interaction.learner[k][2] += 200
    cfg.learner.port += 200
    cfg.actor0.port += 200
    cfg.actor1.port += 200
    return cfg


@pytest.fixture(scope='function')
def setup_actor(setup_config):
    actor = {}
    for k, v in setup_config.items():
        if 'actor' in k:
            actor[k] = create_comm_actor(v)
            actor[k].start()
    yield actor
    for a in actor.values():
        a.close()


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
            assert 'data_id' in data.keys()
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
def setup_learner(setup_config):
    cfg = setup_config.coordinator.interaction.learner
    learner = {}
    for _, (name, host, port) in cfg.items():
        learner[name] = Learner(host, port)
        learner[name].start()
    yield learner
    for l in learner.values():
        l.close()


@pytest.mark.unittest
class TestActorWithCoordinator:

    def test_naive(self, setup_config, setup_actor, setup_learner):
        os.popen('rm -rf {}*'.format(DATA_PREFIX))
        os.popen('rm -rf env_*_*')
        os.popen('rm -rf test.pth')
        assert len(setup_actor) == len(setup_config.coordinator.interaction.actor)
        try:
            coordinator = Coordinator(setup_config.coordinator)
            coordinator.start()
            while True:
                if hasattr(setup_actor['actor0'], '_actor'):
                    break
                time.sleep(0.5)
            torch.save(
                {
                    'model': setup_actor['actor0']._actor.policy.state_dict_handle()['model'].state_dict(),
                    'iter': 0
                }, 'test.pth'
            )
            while True:
                commander = coordinator._commander
                if commander._learner_task_finish_count == 1 and commander._actor_task_finish_count == 2:
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

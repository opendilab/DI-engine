from ding.worker.adapter.learner_aggregator import LearnerAggregator
from typing import Union
import numpy as np
import pytest
from easydict import EasyDict

from ding.interaction import Master, Slave, TaskFail
from ding.interaction.master.task import TaskStatus
from ding.utils import build_logger


class LearnerSlave(Slave):

    def __init__(self, id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = 32
        self.learner_step = np.random.randint(100 * id, 100 * id + 100)
        self.buffer_id = "buffer_" + str(np.random.randint(10 * id, 10 * id + 10))
        self.task_id = "task_" + str(np.random.randint(10 * id, 10 * id + 10))
        self.learner_done = True if np.random.rand() < 0.5 else False

    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        task_name = task['name']
        if task_name == 'resource':
            return {'gpu': 1}
        elif task_name == 'learner_start_task':
            return {'message': 'learner task has started'}
        elif task_name == 'learner_get_data_task':
            return {'batch_size': self.batch_size}
        elif task_name == 'learner_learn_task':
            return {
                'learner_step': self.learner_step,
                'buffer_id': self.buffer_id,
                'task_id': self.task_id,
                'learner_done': self.learner_done,
                'a_list': [1, 2],
            }
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal learner task <{}>'.format(task_name))


@pytest.mark.unittest
def test_learner_aggregator():
    learner_slaves = [LearnerSlave(i, '0.0.0.0', 19900 + i) for i in range(4)]
    for learner_slave in learner_slaves:
        learner_slave.start()
    la_cfg = EasyDict(
        master=dict(
            host='0.0.0.0',
            port=19999,
        ),
        slave=dict(
            host='0.0.0.0',
            port=18800,
        ),
        learner=dict(
            learner0=('learner0', '0.0.0.0', 19900),
            learner1=('learner1', '0.0.0.0', 19901),
            learner2=('learner2', '0.0.0.0', 19902),
            learner3=('learner3', '0.0.0.0', 19903),
        )
    )
    learner_aggregator = LearnerAggregator(la_cfg)
    learner_aggregator.start()
    with Master('0.0.0.0', 18888) as master:  # coordinator master
        master.ping()  # True if master launch success, otherwise False
        with master.new_connection('with_la_slave', '0.0.0.0', 18800) as conn:
            assert conn.is_connected
            assert 'with_la_slave' in master

            task = conn.new_task({'name': 'resource'})
            task.start().join()
            assert task.result == {'gpu': 4}
            assert task.status == TaskStatus.COMPLETED

            task = conn.new_task({'name': 'learner_start_task', 'task_info': {}})
            task.start().join()
            assert task.result == {'message': 'learner task has started'}
            assert task.status == TaskStatus.COMPLETED

            task = conn.new_task({'name': 'learner_get_data_task', 'task_info': {}})
            task.start().join()
            sum_batch_size = sum([learner.batch_size for learner in learner_slaves])
            assert task.result['batch_size'] == sum_batch_size
            assert task.status == TaskStatus.COMPLETED

            task = conn.new_task({'name': 'learner_learn_task', 'data': [i for i in range(sum_batch_size)]})
            task.start().join()
            assert task.result['learner_step'] == learner_slaves[0].learner_step
            assert task.result['buffer_id'] == learner_slaves[0].buffer_id
            assert task.result['task_id'] == learner_slaves[0].task_id
            assert task.result['learner_done'] == learner_slaves[0].learner_done
            assert task.result['a_list'] == [1, 2] * 4
            assert task.status == TaskStatus.COMPLETED

            task = conn.new_task({'name': 'fake_task', 'task_info': {}})
            task.start().join()
            assert task.status == TaskStatus.FAILED
            assert task.result == {'message': 'task name error'}

            assert learner_aggregator.deal_with_get_resource()['gpu'] == len(learner_slaves)

    learner_aggregator.close()
    for learner_slave in learner_slaves:
        learner_slave.close()

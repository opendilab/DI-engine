import time
import os
from ding.interaction import Slave, TaskFail
from ding.utils import lists_to_dicts


class NaiveLearner(Slave):

    def __init__(self, *args, prefix='', **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = prefix

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
            return {
                'task_id': self.task_info['task_id'],
                'buffer_id': self.task_info['buffer_id'],
                'batch_size': 2,
                'cur_learner_iter': 1
            }
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
                    'learner_step': self.count
                },
                'task_id': self.task_info['task_id'],
                'buffer_id': self.task_info['buffer_id']
            }
            ret['info']['priority_info'] = {k: data[k] for k in priority_keys}
            if self.count > 5:
                ret['info']['learner_done'] = True
                os.popen('touch {}_final_model.pth'.format(self._prefix))
            return ret
        elif task_name == 'learner_close_task':
            return {'task_id': self.task_info['task_id'], 'buffer_id': self.task_info['buffer_id']}
        else:
            raise TaskFail(
                result={'message': 'task name error'}, message='illegal collector task <{}>'.format(task_name)
            )

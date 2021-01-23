import time
import os
from nervex.interaction import Slave, TaskFail
from nervex.utils import lists_to_dicts

DATA_PREFIX = 'SLAVE_ACTOR_DATA'


class NaiveLearner(Slave):

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

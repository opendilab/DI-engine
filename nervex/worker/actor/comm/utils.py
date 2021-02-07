import torch
from nervex.interaction.slave import Slave, TaskFail

DATA_PREFIX = 'SLAVE_ACTOR_DATA'


class NaiveActor(Slave):
    """
    Overview:
        A slave, whose master is coordinator.
        Used to pass message between comm actor and coordinator.
    Interfaces:
        _process_task, _get_timestep
    """

    def _process_task(self, task):
        """
        Overview:
            Process a task according to input task info dict, which is passed in by master coordinator.
            For each type of task, you can refer to corresponding callback function in comm actor for details.
        Arguments:
            - cfg (:obj:`EasyDict`): Task dict. Must contain key "name".
        Returns:
            - result (:obj:`Union[dict, TaskFail]`): Task result dict, or task fail exception.
        """
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
            data = {'data_id': data_id, 'buffer_id': self.task_info['buffer_id'], 'unroll_split_begin': 0}
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

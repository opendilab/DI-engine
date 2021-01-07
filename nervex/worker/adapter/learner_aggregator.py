from typing import Union, Optional
import copy
from nervex.interaction import Master, Slave, TaskFail
from nervex.interaction.master.task import TaskStatus


class LearnerAggregatorSlave(Slave):
    def __init__(self, *args, callback_fn: Optional[dict] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._callback_fn = callback_fn

    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        task_name = task['name']
        if task_name == 'resource':
            return self._callback_fn['deal_with_get_resource']()
        elif task_name == 'learner_start_task':
            return self._callback_fn['deal_with_learner_start'](task)
        elif task_name == 'learner_get_data_task':
            return self._callback_fn['deal_with_get_data']()
        elif task_name == 'learner_learn_task':
            return self._callback_fn['deal_with_learn']()
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))


class LearnerAggregator(object):
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        callback_fn = None
        host, port = cfg.host, cfg.port
        self._slave = LearnerAggregatorSlave(host, port, callback_fn)

        self._world_size = 0
        self._learner_connection = {}

    def start(self) -> None:
        self._slave.start()

    def close(self) -> None:
        self._slave.close()

    def deal_with_get_resource(self) -> dict:
        return {'gpu': self._world_size}

    def deal_with_learner_start(self, task: dict) -> dict:
        if len(self._learner_connection) == 0:
            raise TaskFail(message='no connected learner')
        name = task['name']
        start_task = {}
        for k, v in self._learner_connection.items():
            start_task[k] = v.new_task({'name': name, 'task_info': task['task_info']})
            start_task[k].start()
        for k, v in start_task.items():
            v.join()
        task_status = [v.status for v in start_task.values()]
        if any([s != TaskStatus.COMPLETED for s in task_status]):
            # TODO(nyz) dynamic learner gpu add/remove
            raise TaskFail(message="one of learner can't start task")
        return {'message': 'learner task has started'}

    def deal_with_get_data(self, task: dict) -> dict:
        data_task = {}
        for k, v in self._learner_connection.items():
            data_task[k] = v.new_task({'name': task['name']})
            data_task[k].start()
        for k, v in data_task.items():
            v.join()
        self._data_demand = {k: v.result for k, v in data_task.items()}
        demand_list = list(self._data_demand.values())
        merged_demand = copy.deepcopy(demand_list[0])
        merged_demand['batch_size'] = sum([d['batch_size'] for d in demand_list])
        return merged_demand

    def deal_with_learn(self, task: dict) -> dict:
        learn_task = {}
        merged_data = task['data']
        split_data = []
        start = 0
        for item in self._data_demand.values():
            end = item['batch_size'] + start
            split_data.append(merged_data[start:end])
            start = end
        for (k, v), d in zip(self._learner_connection.items(), split_data):
            learn_task[k] = v.new_task({'name': task['name'], 'data': d})
            learn_task.start()
        for k, v in learn_task.items():
            v.join()
        info_list = [v.result for v in learn_task.values()]
        merged_info = self.merge_info(info_list)
        return merged_info

    @staticmethod
    def merge_info(info_list: list) -> dict:
        pass

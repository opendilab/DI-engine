from typing import Union, Optional
import traceback
import numbers
import copy
import time
from functools import reduce
from nervex.interaction import Master, Slave, TaskFail
from nervex.interaction.master.task import TaskStatus
from nervex.utils import build_logger


class LearnerAggregatorSlave(Slave):
    """
    Overview:
        A slave, whose master is coordinator.
    """

    def __init__(self, *args, callback_fn: Optional[dict] = None, **kwargs) -> None:
        """
        Overview:
            Init callback functions additionally. Callback functions are methods in ``LearnerAggregator``.
            As for callback mechanisim, you can refer to ``worker/learner/comm/flask_fs_learner.py`` for help.
        """
        super().__init__(*args, **kwargs)
        self._callback_fn = callback_fn

    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        """
        Overview:
            Process a task according to input task info dict, which is passed in by coordinator's master.
            For each type of task, you can refer to corresponding callback function in
            ``LearnerAggregator`` for details.
        Arguments:
            - cfg (:obj:`EasyDict`): Task dict. Must contain key "name".
        Returns:
            - result (:obj:`Union[dict, TaskFail]`): Task result dict, or task fail exception.
        """
        task_name = task['name']
        if task_name == 'resource':
            return self._callback_fn['deal_with_get_resource']()
        elif task_name == 'learner_start_task':
            return self._callback_fn['deal_with_learner_start'](task)
        elif task_name == 'learner_get_data_task':
            return self._callback_fn['deal_with_get_data'](task)
        elif task_name == 'learner_learn_task':
            return self._callback_fn['deal_with_learn'](task)
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal learner task <{}>'.format(task_name))


class LearnerAggregator(object):
    """
    Overview:
        Aggregate todo duojiduoka danjiduoka
    Interfaces:
        __init__, start, close, merge_info
    """

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Init method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict.
        """
        self._cfg = cfg
        callback_fn = {
            'deal_with_get_resource': self.deal_with_get_resource,
            'deal_with_learner_start': self.deal_with_learner_start,
            'deal_with_get_data': self.deal_with_get_data,
            'deal_with_learn': self.deal_with_learn,
        }
        host, port = cfg.slave.host, cfg.slave.port
        self._slave = LearnerAggregatorSlave(host, port, callback_fn=callback_fn)
        self._logger, _ = build_logger(path='./log', name='learner_aggregator', need_tb=False)

        # ``_world_size`` indicates how many learners are connected;
        # And ``_learner_connection`` lists those connections in dict type.
        self._world_size = 0
        self._learner_connection = {}

    def start(self) -> None:
        """
        Overview:
            Start the aggregator. Set up a master and build connections with all learners within max retry time.
        """
        try:
            self._slave.start()
        except Exception as e:
            self._logger.error(
                "learner_aggregator slave start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)
            )
            return
        max_retry_time = 60
        start_time = time.time()
        while time.time() - start_time <= max_retry_time:
            try:
                self._master = Master(self._cfg.master.host, self._cfg.master.port)
                self._master.start()
                self._master.ping()
                self._world_size = 0
                for _, (learner_id, learner_host, learner_port) in self._cfg.learner.items():
                    conn = self._master.new_connection(learner_id, learner_host, learner_port)
                    conn.connect()
                    assert conn.is_connected
                    self._logger.info("learner {} is connected".format(learner_id))
                    self._learner_connection[learner_id] = conn
                    self._world_size += 1
                self._logger.info("learner aggregator is started")
                break
            except Exception as e:
                # retry not close slave
                try:
                    for _, conn in self._learner_connection.items():
                        conn.disconnect()
                        assert not conn.is_connected
                    self._learner_connection.clear()
                    self._master.close()
                except Exception:
                    pass
                self._logger.error(
                    "learner_aggregator master start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) +
                    repr(e)
                )
                time.sleep(5)
        # Exceeds max retry time and no learner connection found.
        if len(self._learner_connection) == 0:
            self._logger.error("learner_aggregator master max retries failed")

    def close(self) -> None:
        """
        Overview:
            Close aggregator slave, connections with learners, and master.
        """
        try:
            self._slave.close()
            for _, conn in self._learner_connection.items():
                conn.disconnect()
                assert not conn.is_connected
            self._master.close()
        except Exception:  # Ignore close exception.
            pass

    def deal_with_get_resource(self) -> dict:
        return {'gpu': self._world_size}

    def deal_with_learner_start(self, task: dict) -> dict:
        if len(self._learner_connection) == 0:
            raise TaskFail(message='no connected learner', result={'message': 'no connected learner'})
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
            message = "one of learner can't start_task"
            raise TaskFail(message=message, result={'message': message})
        return {'message': 'learner task has started'}

    def deal_with_get_data(self, task: dict) -> dict:
        data_task = {}
        for k, v in self._learner_connection.items():
            data_task[k] = v.new_task({'name': task['name']})
            data_task[k].start()
        for k, v in data_task.items():
            v.join()
        # TODO deal with task fail
        self._data_demand = {k: v.result for k, v in data_task.items()}
        demand_list = list(self._data_demand.values())
        # Merge data demand info by adding up all learners' demand batch size.
        merged_demand = copy.deepcopy(demand_list[0])
        merged_demand['batch_size'] = sum([d['batch_size'] for d in demand_list])
        return merged_demand

    def deal_with_learn(self, task: dict) -> dict:
        learn_task = {}
        merged_data = task['data']
        # Split training data for each learner according to ``self._data_demand``.
        split_data = []
        start = 0
        for item in self._data_demand.values():
            end = item['batch_size'] + start
            split_data.append(merged_data[start:end])
            start = end
        for (k, v), d in zip(self._learner_connection.items(), split_data):
            learn_task[k] = v.new_task({'name': task['name'], 'data': d})
            learn_task[k].start()
        for k, v in learn_task.items():
            v.join()
        # TODO deal with task fail
        info_list = [v.result for v in learn_task.values()]
        # Merge learn info through ``merge_info`` method.
        merged_info = self.merge_info(info_list)
        return merged_info

    @staticmethod
    def merge_info(info: list) -> dict:
        homogeneous_keys = ['learner_step', 'buffer_id', 'task_id', 'learner_done']
        elem = info[0]
        if elem is None:
            return info
        elif isinstance(elem, numbers.Integral) or isinstance(elem, str) or isinstance(elem, float):
            return info
        elif isinstance(elem, list) or isinstance(elem, tuple):
            return list(reduce(lambda x, y: x + y, info))
        elif isinstance(elem, dict):
            ret = {}
            for k in elem.keys():
                if k in homogeneous_keys:
                    ret[k] = elem[k]
                else:
                    ret[k] = LearnerAggregator.merge_info([e[k] for e in info])
            return ret
        else:
            raise TypeError("not support type: {}".format(type(elem)))

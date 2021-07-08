from typing import Union, Optional
import traceback
import numbers
import copy
import time
from functools import reduce
from threading import Thread
from easydict import EasyDict

from ding.interaction import Master, Slave, TaskFail
from ding.interaction.master.task import TaskStatus
from ding.utils import build_logger, get_operator_server_kwargs, exist_operator_server
from ..coordinator.operator_server import OperatorServer


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
        Aggregate multiple learners.
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
        self._end_flag = True
        self._max_retry_second = 60

        # ``_world_size`` indicates how many learners are connected;
        # And ``_learner_connection`` lists those connections in dict type.
        self._world_size = 0
        self._learner_connection = {}

        # create operator server
        if exist_operator_server():
            # get from default or env vars
            server_kwargs = get_operator_server_kwargs(EasyDict({}))
            self._operator_server = OperatorServer(**server_kwargs)
            self._operator_server.set_worker_type('aggregator')
        else:
            self._operator_server = None

        # failed connection
        self._failed_learner_conn = set()

    def start(self) -> None:
        """
        Overview:
            Start the aggregator. Set up a master and build connections with all learners within max retry time.
        """
        self._end_flag = False
        try:
            self._slave.start()
        except Exception as e:
            self._logger.error(
                "learner_aggregator slave start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)
            )
            return
        try:
            self._master = Master(self._cfg.master.host, self._cfg.master.port)
            self._master.start()
            self._master.ping()
        except Exception as e:
            self._logger.error(
                "learner_aggregator master start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)
            )
            return
        self._world_size = 0
        for _, (learner_id, learner_host, learner_port) in self._cfg.learner.items():
            self._new_connection_learner(learner_id, learner_host, int(learner_port))

        if self._operator_server:
            self._init_conn_flag = False
            # create sync learner thread
            self._period_sync_with_server_thread = Thread(
                target=self._period_sync_with_server, name="period_sync", daemon=True
            )
            self._period_sync_with_server_thread.start()
            start_time = time.time()
            while time.time() - start_time <= self._max_retry_second and not self._end_flag:
                if not self._init_conn_flag:
                    time.sleep(0.2)

        # Exceeds max retry time and no learner connection found.
        if len(self._learner_connection) == 0:
            self._logger.error("learner_aggregator master max retries failed")
        else:
            self._logger.info("learner aggregator is started")

    def close(self) -> None:
        """
        Overview:
            Close aggregator slave, connections with learners, and master.
        """
        if self._end_flag:
            return
        self._end_flag = True
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

    def _new_connection_learner(self, learner_id: str, learner_host: str, learner_port: int) -> None:
        start_time = time.time()
        conn = None
        while time.time() - start_time <= self._max_retry_second and not self._end_flag:
            try:
                if conn is None or not conn.is_connected:
                    conn = self._master.new_connection(learner_id, learner_host, learner_port)
                    conn.connect()
                    assert conn.is_connected
                    self._learner_connection[learner_id] = conn
                    self._world_size += 1
                    break
            except Exception as e:
                self._logger.error(
                    f"learner({learner_id}) connection start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) +
                    repr(e) + '\nAuto Retry...'
                )
                time.sleep(2)

        if learner_id in self._learner_connection:
            self._logger.info(f"Succeed to connect to learner({learner_id})")
        else:
            self._logger.info(f"Fail to connect to learner({learner_id})")
            self._failed_learner_conn.add(learner_id)

    def _update_connection_learner(self, cur_learners) -> None:
        conn_learners = list(self._learner_connection.keys())
        new_c = set(cur_learners) - set(conn_learners)
        del_c = set(conn_learners) - (set(cur_learners) | self._failed_learner_conn)
        # conns which have terminated in server side, clear up
        self._failed_learner_conn = self._failed_learner_conn & set(cur_learners)

        # connect to each new learner
        for learner_id in new_c:
            learner_host, learner_port = learner_id.split(':')
            self._new_connection_learner(learner_id, learner_host, int(learner_port))

        for learner_id in del_c:
            if learner_id in conn_learners:
                if self._connection_learner[learner_id].is_connected:
                    conn = self._connection_learner.pop(learner_id)
                    conn.disconnect()
                    assert not conn.is_connected
                else:
                    # ignore the operation of disconnect, since the pod will be terminated by server,
                    # just throw the connection
                    self._connection_learner.pop(learner_id)

    def _period_sync_with_server(self) -> None:
        while not self._end_flag:
            # First: send failed list to notify server which replicas are failed, then terminate such replicas.
            if len(self._failed_learner_conn) > 0:
                learner_conn = []
                for replica_conn in self._failed_learner_conn:
                    dns_name = replica_conn.split(":")[0]
                    pod_name_list = dns_name.split(".")[:-1]
                    pod_name = ".".join(pod_name_list)
                    if pod_name not in learner_conn:
                        learner_conn.append(pod_name)
                success, _, message, _ = self._operator_server.post_replicas_failed(learners=list(learner_conn))
                if success:
                    # do not update learner instantly, update at /GET replicas
                    self._failed_learner_conn.clear()
                else:
                    self._logger.error("Failed to send failed list to server, message: {}".format(message))

            # get list from server
            success, _, message, data = self._operator_server.get_replicas()
            if success:
                cur_learners = data["learners"]
                # self._logger.info("current list:", cur_learners)
                self._update_connection_learner(cur_learners)
                self._init_conn_flag = self._init_conn_flag | True
            else:
                self._logger.error("Failed to sync with server, message: {}".format(message))

            time.sleep(3)

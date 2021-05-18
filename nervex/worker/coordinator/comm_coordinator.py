import traceback
import time
import sys
import requests
from typing import Dict, Callable
from threading import Thread

from nervex.utils import LockContext, LockContextType, get_operator_server_kwargs
from nervex.interaction import Master, OperatorServer
from nervex.interaction.master.task import TaskStatus
from .resource_manager import NaiveResourceManager


class CommCoordinator(object):
    r"""
    Overview:
        the communication part of coordinator(coordinator intercollector)
    Interface:
        __init__ , start, close, __del__, send_collector_task, send_learner_task
    """

    def __init__(self, cfg: dict, callback_fn: Dict[str, Callable], logger: 'TextLogger') -> None:  # noqa
        r"""
        Overview:
            init the interactor of coordinator
        Arguments:
            - cfg (:obj:`dict`): the config file of coordinator interactor
            - callback_fn (:obj:`Dict[str, Callable]`): the callback functions given by coordinator
            - logger (:obj:`TextLogger`): the logger
        """
        self._cfg = cfg
        self._callback_fn = callback_fn
        self._logger = logger
        self._max_retry_second = 120
        self._end_flag = True

        self._connection_collector = {}
        self._connection_learner = {}
        self._connection_lock = LockContext(LockContextType.THREAD_LOCK)
        self._resource_manager = NaiveResourceManager()

        self._remain_task_lock = LockContext(LockContextType.THREAD_LOCK)
        self._remain_collector_task = set()
        self._remain_learner_task = set()

        if self._cfg.operator_server:
            server_kwargs = get_operator_server_kwargs(self._cfg.operator_server)
            self._operator_server = OperatorServer()
        else:
            self._operator_server = None

        # for update resource
        self._resource_lock = LockContext(LockContextType.THREAD_LOCK)

        # failed connection
        self._failed_learner_conn = set()
        self._failed_collector_conn = set()

    def start(self) -> None:
        r"""
        Overview:
            start the coordinator interactor and manage resources and connections
        """
        self._end_flag = False
        self._master = Master(self._cfg.host, self._cfg.port)
        self._master.start()
        self._master.ping()

        # new connection from config
        for _, (learner_id, learner_host, learner_port) in self._cfg.learner.items():
            self._new_connection_learner(learner_id, learner_host, learner_port)
        for _, (collector_id, collector_host, collector_port) in self._cfg.collector.items():
            self._new_connection_collector(collector_id, collector_host, collector_port)

        # create sync learner/collector thread
        if self._operator_server:
            self._period_sync_with_server_thread = Thread(
                target=self._period_sync_with_server, name="period_sync", daemon=True
            )
            self._period_sync_with_server_thread.start()

        if self._end_flag:
            self._logger.error("connection max retries failed")
            sys.exit(1)

    def _new_connection_collector(self, collector_id: str, collector_host: str, collector_port: int) -> None:
        start_time = time.time()
        conn = None
        while time.time() - start_time <= self._max_retry_second and not self._end_flag:
            try:
                if conn is None or not conn.is_connected:
                    conn = self._master.new_connection(collector_id, collector_host, collector_port)
                    conn.connect()
                    assert conn.is_connected
                resource_task = self._get_resource(conn)
                if resource_task.status != TaskStatus.COMPLETED:
                    self._logger.error("can't acquire resource for collector({})".format(collector_id))
                    continue
                else:
                    with self._resource_lock:
                        self._resource_manager.update('collector', collector_id, resource_task.result)
                    with self._connection_lock:
                        self._connection_collector[collector_id] = conn
                    break

            except Exception as e:
                self._logger.error(
                    f"Collector({collector_id}) connection start error:\n" +
                    ''.join(traceback.format_tb(e.__traceback__)) + repr(e) + '\nAuto Retry...'
                )
                time.sleep(2)

        with self._connection_lock:
            if collector_id in self._connection_collector:
                self._logger.info(f"Succeed to connect to collector({collector_id})")
            else:
                self._logger.info(f"Fail to connect to collector({collector_id})")
                self._failed_collector_conn.add(collector_id)

    def _new_connection_learner(self, learner_id: str, learner_host: str, learner_port: int) -> None:
        start_time = time.time()
        conn = None
        while time.time() - start_time <= self._max_retry_second and not self._end_flag:
            try:
                if conn is None or not conn.is_connected:
                    conn = self._master.new_connection(learner_id, learner_host, learner_port)
                    conn.connect()
                    assert conn.is_connected
                resource_task = self._get_resource(conn)
                if resource_task.status != TaskStatus.COMPLETED:
                    self._logger.error("can't acquire resource for learner({})".format(learner_id))
                    continue
                else:
                    with self._resource_lock:
                        self._resource_manager.update('learner', learner_id, resource_task.result)
                    with self._connection_lock:
                        self._connection_learner[learner_id] = conn
                    break

            except Exception as e:
                self._logger.error(
                    f"learner({learner_id}) connection start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) +
                    repr(e) + '\nAuto Retry...'
                )
                time.sleep(2)

        with self._connection_lock:
            if learner_id in self._connection_learner:
                self._logger.info(f"Succeed to connect to learner({learner_id})")
            else:
                self._logger.info(f"Fail to connect to learner({learner_id})")
                self._failed_learner_conn.add(learner_id)

    def close(self) -> None:
        r"""
        Overview:
            close the coordinator interactor
        """
        if self._end_flag:
            return
        self._end_flag = True
        # wait for execute thread
        start_time = time.time()
        while time.time() - start_time <= 60:
            if len(self._remain_learner_task) == 0 and len(self._remain_collector_task) == 0:
                break
            else:
                time.sleep(1)
        for collector_id, conn in self._connection_collector.items():
            conn.disconnect()
            assert not conn.is_connected
        for learner_id, conn in self._connection_learner.items():
            conn.disconnect()
            assert not conn.is_connected
        self._master.close()

    def __del__(self) -> None:
        r"""
        Overview:
            __del__ method will close the coordinator interactor
        """
        self.close()

    def _get_resource(self, conn: 'Connection') -> 'TaskResult':  # noqa
        r"""
        Overview:
            get the resources according to connection
        Arguments:
            - conn (:obj:`Connection`): the connection to get resource_task
        """
        resource_task = conn.new_task({'name': 'resource'})
        resource_task.start().join()
        return resource_task

    def send_collector_task(self, collector_task: dict) -> bool:
        r"""
        Overview:
            send the collector_task to collector_task threads and execute
        Arguments:
            - collector_task (:obj:`dict`): the collector_task to send
        """
        # assert not self._end_flag, "please start interaction first"
        task_id = collector_task['task_id']
        # according to resource info, assign task to a specific collector and adapt task
        assigned_collector = self._resource_manager.assign_collector(collector_task)
        if assigned_collector is None:
            self._logger.error("collector task({}) doesn't have enough collector to execute".format(task_id))
            return False
        collector_task.update(assigned_collector)

        collector_id = collector_task['collector_id']
        start_task = self._connection_collector[collector_id].new_task(
            {
                'name': 'collector_start_task',
                'task_info': collector_task
            }
        )
        start_task.start().join()
        if start_task.status != TaskStatus.COMPLETED:
            self._resource_manager.update(
                'collector', assigned_collector['collector_id'], assigned_collector['resource_info']
            )
            self._logger.error('collector_task({}) start failed: {}'.format(task_id, start_task.result))
            return False
        else:
            self._logger.info('collector task({}) is assigned to collector({})'.format(task_id, collector_id))
            with self._remain_task_lock:
                self._remain_collector_task.add(task_id)
            collector_task_thread = Thread(
                target=self._execute_collector_task, args=(collector_task, ), name='coordinator_collector_task'
            )
            collector_task_thread.start()
            return True

    def _execute_collector_task(self, collector_task: dict) -> None:
        r"""
        Overview:
            execute the collector task
        Arguments:
            - collector_task (:obj:`dict`): the collector task to execute
        """
        close_flag = False
        collector_id = collector_task['collector_id']
        while not self._end_flag:
            try:
                # data task
                data_task = self._connection_collector[collector_id].new_task({'name': 'collector_data_task'})
                self._logger.info('collector data task begin')
                data_task.start().join()
                self._logger.info('collector data task end')
                if data_task.status != TaskStatus.COMPLETED:
                    # TODO(deal with fail task)
                    self._logger.error('collector data task is failed')
                    continue
                result = data_task.result
                task_id = result.get('task_id', None)
                # data result
                if 'data_id' in result:
                    buffer_id = result.get('buffer_id', None)
                    data_id = result.get('data_id', None)
                    self._callback_fn['deal_with_collector_send_data'](task_id, buffer_id, data_id, result)
                # info result
                else:
                    is_finished = self._callback_fn['deal_with_collector_judge_finish'](task_id, result)
                    if not is_finished:
                        continue
                    # close task
                    self._logger.error('close_task: {}\n{}'.format(task_id, result))
                    close_task = self._connection_collector[collector_id].new_task({'name': 'collector_close_task'})
                    close_task.start().join()
                    if close_task.status != TaskStatus.COMPLETED:
                        # TODO(deal with fail task)
                        self._logger.error('collector close is failed')
                        break
                    result = close_task.result
                    task_id = result.get('task_id', None)
                    self._callback_fn['deal_with_collector_finish_task'](task_id, result)
                    resource_task = self._get_resource(self._connection_collector[collector_id])
                    if resource_task.status == TaskStatus.COMPLETED:
                        self._resource_manager.update('collector', collector_id, resource_task.result)
                    close_flag = True
                    break
            except requests.exceptions.HTTPError as e:
                if self._end_flag:
                    break
                else:
                    raise e

        if not close_flag:
            close_task = self._connection_collector[collector_id].new_task({'name': 'collector_close_task'})
            close_task.start().join()
        with self._remain_task_lock:
            self._remain_collector_task.remove(task_id)

    def send_learner_task(self, learner_task: dict) -> bool:
        r"""
        Overview:
            send the learner_task to learner_task threads and execute
        Arguments:
            - learner_task (:obj:`dict`): the learner_task to send
        """
        # assert not self._end_flag, "please start interaction first"
        task_id = learner_task['task_id']
        assigned_learner = self._resource_manager.assign_learner(learner_task)
        if assigned_learner is None:
            self._logger.error("learner task({}) doesn't have enough learner to execute".format(task_id))
            return False
        learner_task.update(assigned_learner)

        learner_id = learner_task['learner_id']
        start_task = self._connection_learner[learner_id].new_task(
            {
                'name': 'learner_start_task',
                'task_info': learner_task
            }
        )
        start_task.start().join()
        if start_task.status != TaskStatus.COMPLETED:
            self._resource_manager.update('learner', assigned_learner['learner_id'], assigned_learner['resource_info'])
            self._logger.info('learner_task({}) start failed: {}'.format(task_id, start_task.result))
            return False
        else:
            self._logger.info('learner task({}) is assigned to learner({})'.format(task_id, learner_id))
            with self._remain_task_lock:
                self._remain_learner_task.add(task_id)
            learner_task_thread = Thread(
                target=self._execute_learner_task, args=(learner_task, ), name='coordinator_learner_task'
            )
            learner_task_thread.start()
            return True

    def _execute_learner_task(self, learner_task: dict) -> None:
        r"""
        Overview:
            execute the learner task
        Arguments:
            - learner_task (:obj:`dict`): the learner task to execute
        """
        close_flag = False
        learner_id = learner_task['learner_id']
        while not self._end_flag:
            try:
                # get data
                get_data_task = self._connection_learner[learner_id].new_task({'name': 'learner_get_data_task'})
                get_data_task.start().join()
                if get_data_task.status != TaskStatus.COMPLETED:
                    # TODO(deal with fail task)
                    self._logger.error('learner get_data_task failed: {}'.format(get_data_task.result))
                    continue
                result = get_data_task.result
                task_id, buffer_id, batch_size = result['task_id'], result['buffer_id'], result['batch_size']
                cur_learner_iter = result['cur_learner_iter']
                sleep_count = 1
                while True:
                    data = self._callback_fn['deal_with_learner_get_data'](
                        task_id, buffer_id, batch_size, cur_learner_iter
                    )
                    if self._end_flag or data is not None:
                        self._logger.info('sample result is ok')
                        break
                    else:
                        self._logger.info('sample result is None')
                        time.sleep(sleep_count)
                        sleep_count += 2
                if self._end_flag:
                    break

                # learn task
                learn_task = self._connection_learner[learner_id].new_task({'name': 'learner_learn_task', 'data': data})
                learn_task.start().join()
                if learn_task.status != TaskStatus.COMPLETED:
                    # TODO(deal with fail task)
                    self._logger.error('learner learn_task failed: {}'.format(learn_task.result))
                    continue
                result = learn_task.result
                task_id, info = result['task_id'], result['info']
                is_finished = self._callback_fn['deal_with_learner_judge_finish'](task_id, info)
                if is_finished:
                    # close task and update resource
                    close_task = self._connection_learner[learner_id].new_task({'name': 'learner_close_task'})
                    close_task.start().join()
                    if close_task.status != TaskStatus.COMPLETED:
                        self._logger.error('learner close_task failed: {}'.format(close_task.result))
                        break
                    result = close_task.result
                    task_id = result.get('task_id', None)
                    self._callback_fn['deal_with_learner_finish_task'](task_id, result)
                    resource_task = self._get_resource(self._connection_learner[learner_id])
                    if resource_task.status == TaskStatus.COMPLETED:
                        self._resource_manager.update('learner', learner_id, resource_task.result)
                    close_flag = True
                    break
                else:
                    # update info
                    buffer_id = result['buffer_id']
                    self._callback_fn['deal_with_learner_send_info'](task_id, buffer_id, info)
            except requests.exceptions.HTTPError as e:
                if self._end_flag:
                    break
                else:
                    raise e

        if not close_flag:
            close_task = self._connection_learner[learner_id].new_task({'name': 'learner_close_task'})
            close_task.start().join()
        with self._remain_task_lock:
            self._remain_learner_task.remove(task_id)

    def _period_sync_with_server(self) -> None:
        start_time = time.time()
        while not self._end_flag:
            # First: send failed list to notify nerveX-server which replicas are failed, then terminate such replicas.
            # self._logger.info("failed list:", list(self._failed_collector_conn), list(self._failed_learner_conn))
            if len(self._failed_learner_conn) > 0 or len(self._failed_collector_conn) > 0:
                success, _, message, _ = self._operator_server.post_replicas_failed(
                    learners=list(self._failed_learner_conn), collectors=list(self._failed_collector_conn)
                )
                if success:
                    # do not update collector or learner instantly, update at /GET replicas
                    self._failed_collector_conn.clear()
                    self._failed_learner_conn.clear()
                else:
                    self._logger.error("Failed to send failed list to server, message: {}".format(message))

            # get list from server
            success, _, message, data = self._operator_server.get_replicas()
            if success:
                cur_collectors = data["collectors"]
                cur_learners = data["learners"]
                # self._logger.info("current list:", cur_collectors, cur_learners)
                self._update_connection_collector(cur_collectors)
                self._update_connection_learner(cur_learners)
            else:
                self._logger.error("Failed to sync with server, message: {}".format(message))

            time.sleep(2)

    def _update_connection_collector(self, cur_collectors) -> None:
        raise NotImplementedError

    def _update_connection_learner(self, cur_learners) -> None:
        raise NotImplementedError

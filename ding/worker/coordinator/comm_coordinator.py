import traceback
import time
import sys
import requests
from typing import Dict, Callable
from threading import Thread

from ding.utils import LockContext, LockContextType, get_operator_server_kwargs
from ding.interaction import Master
from ding.interaction.master.task import TaskStatus
from .resource_manager import NaiveResourceManager
from .operator_server import OperatorServer


class CommCoordinator(object):
    r"""
    Overview:
        the communication part of coordinator(coordinator intercollector)
    Interface:
        __init__ , start, close, __del__, send_collector_task, send_learner_task
    """

    def __init__(self, cfg: dict, callback_fn: Dict[str, Callable], logger: 'logging.Logger') -> None:  # noqa
        r"""
        Overview:
            init the interactor of coordinator
        Arguments:
            - cfg (:obj:`dict`): The config file of communication coordinator
            - callback_fn (:obj:`Dict[str, Callable]`): The callback functions given by coordinator
            - logger (:obj:`logging.Logger`): The text logger.
        """
        self._cfg = cfg
        self._callback_fn = callback_fn
        self._logger = logger
        self._max_retry_second = 120
        self._end_flag = True

        self._connection_collector = {}
        self._connection_learner = {}
        self._resource_manager = NaiveResourceManager()

        self._remain_task_lock = LockContext(LockContextType.THREAD_LOCK)
        self._remain_collector_task = set()
        self._remain_learner_task = set()

        if self._cfg.operator_server:
            server_kwargs = get_operator_server_kwargs(self._cfg.operator_server)
            self._operator_server = OperatorServer(**server_kwargs)
            self._operator_server.set_worker_type('coordinator')
            self._collector_target_num = self._cfg.operator_server.collector_target_num
            self._learner_target_num = self._cfg.operator_server.learner_target_num
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

        if self._operator_server:
            # post init learner/collector demand
            start_time, init_flag = time.time(), False
            while time.time() - start_time <= self._max_retry_second and not self._end_flag:
                success, _, message, _ = self._operator_server.post_replicas(
                    self._cfg.operator_server.init_replicas_request
                )
                if success:
                    self._logger.info("Post replicas demand to server successfully")
                    init_flag = True
                    break
                else:
                    self._logger.info("Failed to post replicas request to server, message: {}".format(message))
                    time.sleep(2)

            if not init_flag:
                self._logger.info('Exit since cannot request replicas to operator-server...')
                self.close()
                sys.exit(1)

            # create sync learner/collector thread
            self._period_sync_with_server_thread = Thread(
                target=self._period_sync_with_server, name="period_sync", daemon=True
            )
            self._period_sync_with_server_thread.start()

            # wait for enough collector/learner
            start_time = time.time()
            enough_flag = False
            while time.time() - start_time <= self._max_retry_second:
                if len(self._connection_collector) < self._collector_target_num and len(self._connection_learner
                                                                                        ) < self._learner_target_num:
                    self._logger.info(
                        "Only can connect {} collectors, {} learners.".format(
                            len(self._connection_collector), len(self._connection_learner)
                        )
                    )
                    time.sleep(2)
                else:
                    self._logger.info(
                        "Have connected {} collectors, {} learners, match limit requests.".format(
                            len(self._connection_collector), len(self._connection_learner)
                        )
                    )
                    self._logger.info("Total DI-engine pipeline start...")
                    enough_flag = True
                    break

            if not enough_flag:
                self._logger.error(
                    "Exit since only can connect {} collectors, {} learners.".format(
                        len(self._connection_collector), len(self._connection_learner)
                    )
                )
                self.close()
                sys.exit(1)

        if self._end_flag:
            self._logger.error("connection max retries failed")
            sys.exit(1)

    def _new_connection_collector(
            self,
            collector_id: str,
            collector_host: str,
            collector_port: int,
            increase_task_space: bool = False,
    ) -> None:
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
                    self._connection_collector[collector_id] = conn
                    if increase_task_space:
                        self._callback_fn['deal_with_increase_collector']()
                    break

            except Exception as e:
                self._logger.error(
                    f"Collector({collector_id}) connection start error:\n" +
                    ''.join(traceback.format_tb(e.__traceback__)) + repr(e) + '\nAuto Retry...'
                )
                time.sleep(2)

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
                    self._connection_learner[learner_id] = conn
                    break

            except Exception as e:
                self._logger.error(
                    f"learner({learner_id}) connection start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) +
                    repr(e) + '\nAuto Retry...'
                )
                time.sleep(2)

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
        # TODO
        if self._operator_server:
            self._period_sync_with_server_thread.join()
            # wait from all slave receive DELETE
            time.sleep(5)
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
        while not self._end_flag:
            # First: send failed list to notify DI-engine server which replicas are failed,
            # then terminate such replicas.
            # self._logger.info("failed list:", list(self._failed_collector_conn), list(self._failed_learner_conn))
            if len(self._failed_learner_conn) > 0 or len(self._failed_collector_conn) > 0:
                collector_conn = []
                for replica_conn in self._failed_collector_conn:
                    dns_name = replica_conn.split(":")[0]
                    pod_name_list = dns_name.split(".")[:-1]
                    pod_name = ".".join(pod_name_list)
                    collector_conn.append(pod_name)
                learner_conn = []
                for replica_conn in self._failed_learner_conn:
                    dns_name = replica_conn.split(":")[0]
                    pod_name_list = dns_name.split(".")[:-1]
                    pod_name = ".".join(pod_name_list)
                    learner_conn.append(pod_name)

                success, _, message, _ = self._operator_server.post_replicas_failed(
                    learners=list(learner_conn), collectors=list(collector_conn)
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

            time.sleep(1)

    def _update_connection_collector(self, cur_collectors: list) -> None:
        conn_collectors = list(self._connection_collector.keys())
        new_c = set(cur_collectors) - set(conn_collectors)
        del_c = set(conn_collectors) - (set(cur_collectors) | self._failed_collector_conn)
        # conns which have terminated in server side, clear up
        self._failed_collector_conn = self._failed_collector_conn & set(cur_collectors)

        # connect to each new collector
        for collector_id in new_c:
            collector_host, collector_port = collector_id.split(':')
            self._new_connection_collector(collector_id, collector_host, int(collector_port), True)

        for collector_id in del_c:
            if collector_id in conn_collectors:
                # TODO(nyz) whether to need to close task first
                with self._resource_lock:
                    if not self._resource_manager.have_assigned('collector', collector_id):
                        self._resource_manager.delete("collector", collector_id)

                if self._connection_collector[collector_id].is_connected:
                    conn = self._connection_collector.pop(collector_id)
                    conn.disconnect()
                    assert not conn.is_connected
                    self._callback_fn['deal_with_decrease_collector']()
                else:
                    # ignore the operation of disconnect, since the pod will be terminated by server,
                    # just throw the connection
                    self._connection_collector.pop(collector_id)

    def _update_connection_learner(self, cur_learners) -> None:
        conn_learners = list(self._connection_learner.keys())
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
                # TODO(nyz) whether to need to close task first
                with self._resource_lock:
                    if not self._resource_manager.have_assigned('learner', learner_id):
                        self._resource_manager.delete("learner", learner_id)

                if self._connection_learner[learner_id].is_connected:
                    conn = self._connection_learner.pop(learner_id)
                    conn.disconnect()
                    assert not conn.is_connected
                else:
                    # ignore the operation of disconnect, since the pod will be terminated by server,
                    # just throw the connection
                    self._connection_learner.pop(learner_id)

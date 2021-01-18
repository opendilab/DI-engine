import traceback
import time
import sys
import requests
from typing import Dict, Callable
from threading import Thread

from nervex.utils import LockContext, LockContextType
from nervex.interaction.master import Master
from nervex.interaction.master.task import TaskStatus
from .resource_manager import NaiveResourceManager


class CoordinatorInteraction(object):

    def __init__(self, cfg: dict, callback_fn: Dict[str, Callable], logger: 'TextLogger') -> None:  # noqa
        self._cfg = cfg
        self._callback_fn = callback_fn
        self._logger = logger
        self._connection_actor = {}
        self._connection_learner = {}
        self._resource_manager = NaiveResourceManager()
        self._end_flag = True
        self._remain_lock = LockContext(LockContextType.THREAD_LOCK)
        self._remain_actor_task = set()
        self._remain_learner_task = set()

    def start(self) -> None:
        max_retry_time = 120
        start_time = time.time()
        while time.time() - start_time <= max_retry_time:
            self._end_flag = False
            self._master = Master(self._cfg.host, self._cfg.port)
            try:
                self._master.start()
                self._master.ping()
                for _, (learner_id, learner_host, learner_port) in self._cfg.learner.items():
                    conn = self._master.new_connection(learner_id, learner_host, learner_port)
                    conn.connect()
                    assert conn.is_connected
                    resource_task = self._get_resource(conn)
                    if resource_task.status != TaskStatus.COMPLETED:
                        self._logger.error("can't acquire resource for learner({})".format(learner_id))
                        continue
                    else:
                        self._resource_manager.update('learner', learner_id, resource_task.result)
                        self._connection_learner[learner_id] = conn
                for _, (actor_id, actor_host, actor_port) in self._cfg.actor.items():
                    conn = self._master.new_connection(actor_id, actor_host, actor_port)
                    conn.connect()
                    assert conn.is_connected
                    resource_task = self._get_resource(conn)
                    if resource_task.status != TaskStatus.COMPLETED:
                        self._logger.error("can't acquire resource for actor({})".format(actor_id))
                        continue
                    else:
                        self._resource_manager.update('actor', actor_id, resource_task.result)
                        self._connection_actor[actor_id] = conn
                break
            except Exception as e:
                self.close()
                self._logger.error(
                    "connection start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) + repr(e) +
                    '\nAuto Retry...'
                )
                time.sleep(5)
        if self._end_flag:
            self._logger.error("connection max retries failed")
            sys.exit(1)

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        # wait for execute thread
        start_time = time.time()
        while time.time() - start_time <= 60:
            if len(self._remain_learner_task) == 0 and len(self._remain_actor_task) == 0:
                break
            else:
                time.sleep(1)
        for actor_id, conn in self._connection_actor.items():
            conn.disconnect()
            assert not conn.is_connected
        for learner_id, conn in self._connection_learner.items():
            conn.disconnect()
            assert not conn.is_connected
        self._master.close()

    def __del__(self) -> None:
        self.close()

    def _get_resource(self, conn: 'Connection') -> 'TaskResult':  # noqa
        resource_task = conn.new_task({'name': 'resource'})
        resource_task.start().join()
        return resource_task

    def send_actor_task(self, actor_task: dict) -> bool:
        # assert not self._end_flag, "please start interaction first"
        task_id = actor_task['task_id']
        # according to resource info, assign task to a specific actor and adapt task
        assigned_actor = self._resource_manager.assign_actor(actor_task)
        if assigned_actor is None:
            self._logger.error("actor task({}) doesn't have enough actor to execute".format(task_id))
            return False
        actor_task.update(assigned_actor)

        actor_id = actor_task['actor_id']
        start_task = self._connection_actor[actor_id].new_task({'name': 'actor_start_task', 'task_info': actor_task})
        start_task.start().join()
        if start_task.status != TaskStatus.COMPLETED:
            self._resource_manager.update('actor', assigned_actor['actor_id'], assigned_actor['resource_info'])
            self._logger.error('actor_task({}) start failed: {}'.format(task_id, start_task.result))
            return False
        else:
            self._logger.info('actor task({}) is assigned to actor({})'.format(task_id, actor_id))
            with self._remain_lock:
                self._remain_actor_task.add(task_id)
            actor_task_thread = Thread(target=self._execute_actor_task, args=(actor_task, ))
            actor_task_thread.start()
            return True

    def _execute_actor_task(self, actor_task: dict) -> None:
        actor_id = actor_task['actor_id']
        while not self._end_flag:
            try:
                data_task = self._connection_actor[actor_id].new_task({'name': 'actor_data_task'})
                data_task.start().join()
                if data_task.status != TaskStatus.COMPLETED:
                    # TODO(deal with fail task)
                    self._logger.error('actor data task is failed')
                    continue
                else:
                    result = data_task.result
                    finished_task = result.pop('finished_task', None)
                    if finished_task:
                        # result['finished_task'] is a flag
                        task_id = result.get('task_id', None)
                        self._callback_fn['deal_with_actor_finish_task'](task_id, result)
                        resource_task = self._get_resource(self._connection_actor[actor_id])
                        if resource_task.status == TaskStatus.COMPLETED:
                            self._resource_manager.update('actor', actor_id, resource_task.result)
                        break
                    else:
                        task_id = result.get('task_id', None)
                        buffer_id = result.get('buffer_id', None)
                        data_id = result.get('data_id', None)
                        self._callback_fn['deal_with_actor_send_data'](task_id, buffer_id, data_id, result)
            except requests.exceptions.HTTPError as e:
                if self._end_flag:
                    break
                else:
                    raise e

        with self._remain_lock:
            self._remain_actor_task.remove(task_id)

    def send_learner_task(self, learner_task: dict) -> bool:
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
            with self._remain_lock:
                self._remain_learner_task.add(task_id)
            learner_task_thread = Thread(target=self._execute_learner_task, args=(learner_task, ))
            learner_task_thread.start()
            return True

    def _execute_learner_task(self, learner_task: dict) -> None:
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
                sleep_count = 1
                while True:
                    data = self._callback_fn['deal_with_learner_get_data'](task_id, buffer_id, batch_size)
                    if self._end_flag or data is not None:
                        self._logger.info('sample result is ok')
                        break
                    else:
                        self._logger.info('sample result is None')
                        time.sleep(sleep_count)
                        sleep_count += 2
                if self._end_flag:
                    break

                # learn
                learn_task = self._connection_learner[learner_id].new_task({'name': 'learner_learn_task', 'data': data})
                learn_task.start().join()
                if learn_task.status != TaskStatus.COMPLETED:
                    # TODO(deal with fail task)
                    self._logger.error('learner learn_task failed: {}'.format(learn_task.result))
                    continue
                result = learn_task.result
                task_id, finished_task = result['task_id'], result['finished_task']
                # finish task and update resource
                if finished_task:
                    # result['finished_task'] is a flag
                    self._callback_fn['deal_with_learner_finish_task'](task_id, result)
                    resource_task = self._get_resource(self._connection_learner[learner_id])
                    if resource_task.status == TaskStatus.COMPLETED:
                        self._resource_manager.update('learner', learner_id, resource_task.result)
                    break
                else:
                    # update info
                    buffer_id, info = result['buffer_id'], result['info']
                    self._callback_fn['deal_with_learner_send_info'](task_id, buffer_id, info)
            except requests.exceptions.HTTPError as e:
                if self._end_flag:
                    break
                else:
                    raise e

        with self._remain_lock:
            self._remain_learner_task.remove(task_id)

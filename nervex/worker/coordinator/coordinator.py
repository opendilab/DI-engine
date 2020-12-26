import time
from typing import Dict, Callable
from queue import Queue
from threading import Thread

from nervex.utils import build_logger, LockContext, LockContextType, get_task_uid
from nervex.interaction.master import Master
from nervex.interaction.master.task import TaskStatus


class CoordinatorInteraction(object):

    def __init__(self, cfg: dict, callback_fn: Dict[str, Callable]) -> None:
        self._cfg = cfg
        self._callback_fn = callback_fn
        self._callback_fn_lock = LockContext(LockContextType.THREAD_LOCK)
        self._interaction = Master(cfg.host, cfg.port)
        self._connection_actor = {}

    def start(self) -> None:
        self._end_flag = False
        self._interaction.start()
        self._interaction.ping()
        for _, (actor_id, actor_host, actor_port) in self._cfg.actor.items():
            conn = self._interaction.new_connection(actor_id, actor_host, actor_port)
            conn.connect()
            self._connection_actor[actor_id] = conn
            assert conn.is_connected

    def close(self) -> None:
        self._end_flag = True
        for actor_id, conn in self._connection_actor.items():
            conn.disconnect()
            assert not conn.is_connected
        self._interaction.close()

    def send_actor_task(self, actor_task: dict) -> bool:
        actor_id = actor_task['actor_id']
        start_task = self._connection_actor[actor_id].new_task({'name': 'actor_start_task', 'task_info': actor_task})
        start_task.start().join()
        if start_task.status != TaskStatus.COMPLETED:
            return False
        else:
            actor_task_thread = Thread(target=self._execute_actor_task, args=(actor_task, ))
            actor_task_thread.start()
            return True

    def _execute_actor_task(self, actor_task: dict) -> None:
        actor_id = actor_task['actor_id']
        while not self._end_flag:
            data_task = self._connection_actor[actor_id].new_task({'name': 'actor_data_task'})
            data_task.start().join()
            if data_task.status != TaskStatus.COMPLETED:
                # ignore and retry
                continue
            else:
                result = data_task.result
                with self._callback_fn_lock:
                    finish_task = result.pop('finish_task')
                    if finish_task:
                        self._callback_fn['deal_with_actor_finish_task'](result)
                        break
                    else:
                        self._callback_fn['deal_with_actor_send_data'](result)


class Commander(object):

    def __init__(self) -> None:
        self.actor_task_flag = True

    def get_actor_task(self) -> dict:
        if self.actor_task_flag:
            self.actor_task_flag = False
            return {'task_id': 'task_id1'}
        else:
            return None

    def finish_actor_task(self, finished_task: dict) -> None:
        self.actor_task_flag = True


class ResourceManager(object):

    def assign_actor(self, actor_task: dict) -> dict:
        return {'actor_id': 'test'}

    def update_actor_resource(self, finish_task: dict) -> dict:
        return finish_task


class LearnerState(object):

    def __init__(self, replay_buffer_cfg: dict) -> None:
        pass


class ActorState(object):
    pass


class TaskState(object):

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.start_time = time.time()


class Coordinator(object):

    def __init__(self, cfg: dict) -> None:
        self._coordinator_uid = get_task_uid()
        self._cfg = cfg
        self._actor_task_timeout = cfg.coordinator.actor_task_timeout

        self._callback = {
            'deal_with_actor_send_data': self.deal_with_actor_send_data,
            'deal_with_actor_finish_task': self.deal_with_actor_finish_task,
        }
        self._interaction = CoordinatorInteraction(cfg.coordinator.interaction, callback_fn=self._callback)
        self._learner_task_queue = Queue()
        self._actor_task_queue = Queue()
        self._logger, _ = build_logger(path='./log', name='coordinator')
        self._commander = Commander()
        self._commander_lock = LockContext(LockContextType.THREAD_LOCK)
        self._resource_manager = ResourceManager()
        # ############## Thread #####################
        self._assign_actor_thread = Thread(target=self._assign_actor_task, args=())
        self._assign_learner_thread = Thread(target=self._assign_learner_task, args=())
        self._produce_actor_thread = Thread(target=self._produce_actor_task, args=())
        self._produce_learner_thread = Thread(target=self._produce_learner_task, args=())

        self._learner_state = {}  # str -> LearnerState
        self._actor_state = {}  # str -> ActorState
        self._task_state = {}  # str -> TaskState
        # TODO remove used data
        # TODO load/save state_dict
        self._end_flag = False

    def _assign_actor_task(self) -> None:
        while not self._end_flag:
            time.sleep(0.01)
            # get valid task, abandon timeout task
            if self._actor_task_queue.empty():
                continue
            else:
                actor_task, put_time = self._actor_task_queue.get()
                get_time = time.time()
                if get_time - put_time >= self._actor_task_timeout:
                    self.info(
                        'actor task({}) timeout: [{}, {}, {}/{}]'.format(
                            actor_task['task_id'], get_time, put_time, get_time - put_time, self._actor_task_timeout
                        )
                    )
                    continue
            # according to resource info, assign task to a specific actor and adapt task
            assigned_actor = self._resource_manager.assign_actor(actor_task)
            actor_task.update(assigned_actor)
            while True:
                if self._interaction.send_actor_task(actor_task):
                    self._record_task(actor_task)
                    self.info(
                        'actor task({}) is assigned to actor({})'.format(
                            actor_task['task_id'], assigned_actor['actor_id']
                        )
                    )
                    break
                else:
                    self.error(
                        'actor_task({}) is failed to sent to actor({})'.format(
                            actor_task['task_id'], assigned_actor['actor_id']
                        )
                    )

    def _assign_learner_task(self) -> None:
        # get valid task, abandom timeout task
        # according to resource info, assign task to a specific learner and adapt task
        pass

    def _produce_actor_task(self) -> None:
        while not self._end_flag:
            time.sleep(0.01)
            with self._commander_lock:
                actor_task = self._commander.get_actor_task()
                if actor_task is None:
                    continue
            self._actor_task_queue.put([actor_task, time.time()])

    def _produce_learner_task(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def start(self) -> None:
        self._end_flag = False
        self._interaction.start()
        self._produce_actor_thread.start()
        self._assign_actor_thread.start()
        self._produce_learner_thread.start()
        self._assign_learner_thread.start()

    def close(self) -> None:
        self._end_flag = True
        time.sleep(1)
        self._produce_actor_thread.join()
        self._assign_actor_thread.join()
        self._produce_learner_thread.join()
        self._assign_learner_thread.join()
        self._interaction.close()
        # close replay buffer
        for k, v in self._learner_state.items():
            v.replay_buffer.close()

    def deal_with_actor_send_data(self, data: dict) -> None:
        """
        Necessary Key: task_id, learner_id, data_id
        """
        assert data['task_id'] in self._task_state, data['task_id']
        if data['task_id'] not in self._task_state:
            self.error('actor task({}) not in self._task_state when send data, throw it'.format(data['task_id']))
            return
        if 'learner_id' not in data or data['learner_id'] not in self._learner_state:
            self.error("actor task({}) doesn't) have proper learner_id".format(data['task_id']))
            return
        learner_id = data['learner_id']
        self._learner_state['learner_id'].replay_buffer.push_data(data)
        self.info('actor task({}) send data({})'.format(data['task_id'], data['data_id']))

    def deal_with_actor_finish_task(self, finished_task: dict) -> None:
        """
        Necessary Key: task_id
        """
        if finished_task['task_id'] not in self._task_state:
            self.error('actor task({}) not in self._task_state when finish, throw it'.format(finished_task['task_id']))
            return
        assert finished_task['task_id'] in self._task_state, finished_task['task_id']
        # update actor resource info
        finished_task = self._resource_manager.update_actor_resource(finished_task)
        # finish_task
        self._commander.finish_actor_task(finished_task)
        self._task_state.pop(finished_task['task_id'])
        self.info('actor task({}) is finished'.format(finished_task['task_id']))

    def deal_with_learner_get_data(self):
        pass

    def deal_with_learner_send_info(self):
        # train info
        # data info
        pass

    def deal_with_learner_finish_task(self):
        pass

    def info(self, s: str) -> None:
        self._logger.info('[Coordinator({})]: {}'.format(self._coordinator_uid, s))

    def error(self, s: str) -> None:
        self._logger.error('[Coordinator({})]: {}'.format(self._coordinator_uid, s))

    def _record_task(self, task: dict):
        self._task_state[task['task_id']] = TaskState(task['task_id'])

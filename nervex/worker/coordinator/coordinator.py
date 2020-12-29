import time
import traceback
from typing import Dict, Callable, List
from queue import Queue
from threading import Thread

from nervex.utils import build_logger, LockContext, LockContextType, get_task_uid
from nervex.data import ReplayBuffer
from nervex.interaction.master import Master
from nervex.interaction.master.task import TaskStatus


class CoordinatorInteraction(object):

    def __init__(self, cfg: dict, callback_fn: Dict[str, Callable], logger: 'TextLogger') -> None:  # noqa
        self._cfg = cfg
        self._callback_fn = callback_fn
        self._callback_fn_lock = LockContext(LockContextType.THREAD_LOCK)
        self._logger = logger
        self._interaction = Master(cfg.host, cfg.port)
        self._connection_actor = {}
        self._end_flag = True

    def start(self) -> None:
        self._end_flag = False
        try:
            self._interaction.start()
            self._interaction.ping()
            for _, (actor_id, actor_host, actor_port) in self._cfg.actor.items():
                conn = self._interaction.new_connection(actor_id, actor_host, actor_port)
                conn.connect()
                self._connection_actor[actor_id] = conn
                assert conn.is_connected
        except Exception as e:
            self.close()
            self._logger.error("connection start error:\n" + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        for actor_id, conn in self._connection_actor.items():
            conn.disconnect()
            assert not conn.is_connected
        self._interaction.close()

    def __del__(self) -> None:
        self.close()

    def send_actor_task(self, actor_task: dict) -> bool:
        assert not self._end_flag, "please start interaction first"
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
                        task_id = result.get('task_id', None)
                        self._callback_fn['deal_with_actor_finish_task'](task_id, result)
                        break
                    else:
                        task_id = result.get('task_id', None)
                        learner_id = result.get('buffer_id', None)
                        data_id = result.get('data_id', None)
                        self._callback_fn['deal_with_actor_send_data'](task_id, learner_id, data_id, result)


class Commander(object):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self.actor_task_flag = True
        self.learner_task_flag = True

    def get_actor_task(self) -> dict:
        if self.actor_task_flag:
            self.actor_task_flag = False
            return {'task_id': 'task_id1'}
        else:
            return None

    def get_learner_task(self) -> dict:
        if self.learner_task_flag:
            self.learner_task_flag = False
            return {'task_id': 'learner_task_id1'}
        else:
            return None

    def finish_actor_task(self, task_id: str, finished_task: dict) -> None:
        self.actor_task_flag = True

    def finish_learner_task(self, task_id: str, finished_task: dict) -> None:
        self.learner_task_flag = True

    def get_learner_info(self, task_id: str, info: dict) -> None:
        pass


class ResourceManager(object):

    def assign_actor(self, actor_task: dict) -> dict:
        return {'actor_id': 'test'}

    def update_actor_resource(self, finish_task: dict) -> dict:
        return finish_task


class TaskState(object):

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.start_time = time.time()


class Coordinator(object):

    def __init__(self, cfg: dict) -> None:
        self._coordinator_uid = get_task_uid()
        self._cfg = cfg
        self._actor_task_timeout = cfg.coordinator.actor_task_timeout
        self._learner_task_timeout = cfg.coordinator.learner_task_timeout

        self._callback = {
            'deal_with_actor_send_data': self.deal_with_actor_send_data,
            'deal_with_actor_finish_task': self.deal_with_actor_finish_task,
        }
        self._logger, _ = build_logger(path='./log', name='coordinator')
        self._interaction = CoordinatorInteraction(cfg.coordinator.interaction, self._callback, self._logger)
        self._learner_task_queue = Queue()
        self._actor_task_queue = Queue()
        self._commander = Commander(cfg)
        self._commander_lock = LockContext(LockContextType.THREAD_LOCK)
        self._resource_manager = ResourceManager()
        # ############## Thread #####################
        self._assign_actor_thread = Thread(target=self._assign_actor_task, args=())
        self._assign_learner_thread = Thread(target=self._assign_learner_task, args=())
        self._produce_actor_thread = Thread(target=self._produce_actor_task, args=())
        self._produce_learner_thread = Thread(target=self._produce_learner_task, args=())

        self._replay_buffer = {}
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
        return
        while not self._end_flag:
            time.sleep(0.01)
            if self._learner_task_queue.empty():
                continue
            else:
                learner_task, put_time = self._learner_task_queue.get()
                get_time = time.time()
                if get_time - put_time >= self._learner_task_timeout:
                    self.info(
                        'learner task({}) timeout: [{}, {}, {}/{}]'.format(
                            learner_task['task_id'], get_time, put_time, get_time - put_time, self._learner_task_timeout
                        )
                    )
                    continue
                # create replay_buffer
                buffer_id = learner_task['buffer_id']
                if buffer_id not in self._replay_buffer:
                    replay_buffer_cfg = learner_task.pop('replay_buffer_cfg', {})
                    self._replay_buffer[buffer_id] = ReplayBuffer(replay_buffer_cfg)
                    self._replay_buffer[buffer_id].run()
                    self.info("replay_buffer({}) is created".format(buffer_id))
            assigned_learner = self._resource_manager.assign_learner(learner_task)
            learner_task.update(assigned_learner)
            while True:
                if self._interaction.send_learner_task(learner_task):
                    self._record_task(learner_task)
                    self.info(
                        'learner task({}) is assigned to learner({})'.format(
                            learner_task['task_id'], assigned_learner['learner_id']
                        )
                    )
                    break
                else:
                    self.error(
                        'learner_task({}) is failed to sent to learner({})'.format(
                            learner_task['task_id'], assigned_learner['learner_id']
                        )
                    )

    def _produce_actor_task(self) -> None:
        while not self._end_flag:
            time.sleep(0.01)
            with self._commander_lock:
                actor_task = self._commander.get_actor_task()
                if actor_task is None:
                    continue
            self._actor_task_queue.put([actor_task, time.time()])

    def _produce_learner_task(self) -> None:
        while not self._end_flag:
            time.sleep(0.01)
            with self._commander_lock:
                learner_task = self._commander.get_learner_task()
                if learner_task is None:
                    continue
            self._learner_task_queue.put([learner_task, time.time()])

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
        if self._end_flag:
            return
        self._end_flag = True
        time.sleep(1)
        self._produce_actor_thread.join()
        self._assign_actor_thread.join()
        self._produce_learner_thread.join()
        self._assign_learner_thread.join()
        self._interaction.close()
        # close replay buffer
        for k, v in self._replay_buffer.items():
            v.replay_buffer.close()

    def __del__(self) -> None:
        self.close()

    def deal_with_actor_send_data(self, task_id: str, buffer_id: str, data_id: str, data: dict) -> None:
        if task_id not in self._task_state:
            self.error('actor task({}) not in self._task_state when send data, throw it'.format(task_id))
            return
        if buffer_id not in self._replay_buffer:
            self.error("actor task({}) data({}) doesn't have proper buffer_id({})".format(task_id, data_id, buffer_id))
            return
        self._replay_buffer[buffer_id].push_data(data)
        self.info('actor task({}) send data({})'.format(task_id, data_id))

    def deal_with_actor_finish_task(self, task_id: str, finished_task: dict) -> None:
        if task_id not in self._task_state:
            self.error('actor task({}) not in self._task_state when finish, throw it'.format(task_id))
            return
        # update actor resource info
        finished_task = self._resource_manager.update_actor_resource(finished_task)
        # finish_task
        with self._commander_lock:
            self._commander.finish_actor_task(task_id, finished_task)
        self._task_state.pop(task_id)
        self.info('actor task({}) is finished'.format(task_id))

    def deal_with_learner_get_data(self, task_id: str, buffer_id: str, batch_size: int) -> List[dict]:
        if task_id not in self._task_state:
            self.error("learner task({}) get data doesn't have proper task_id".format(task_id))
            raise RuntimeError(
                "invalid learner task_id({}) for get data, valid learner_id is {}".format(
                    task_id, self._task_state.keys()
                )
            )
        if buffer_id not in self._replay_buffer:
            self.error("learner task({}) get data doesn't have proper buffer_id({})".format(task_id, buffer_id))
            return
        self.info("learner task({}) get data".format(task_id))
        return self._replay_buffer[buffer_id].sample(batch_size)

    def deal_with_learner_send_info(self, task_id: str, buffer_id: str, info: dict) -> None:
        if task_id not in self._task_state:
            self.error("learner task({}) send info doesn't have proper task_id".format(task_id))
            raise RuntimeError(
                "invalid learner task_id({}) for send info, valid learner_id is {}".format(
                    task_id, self._task_state.keys()
                )
            )
        if buffer_id not in self._replay_buffer:
            self.error("learner task({}) send info doesn't have proper buffer_id({})".format(task_id, buffer_id))
            return
        self._replay_buffer[buffer_id].update(info)
        with self._commander_lock:
            self._commander.get_learner_info(task_id, info)
        self.info("learner task({}) send info".format(task_id))

    def deal_with_learner_finish_task(self, task_id: str, finished_task: dict) -> None:
        if task_id not in self._task_state:
            self.error("learner task({}) finish task doesn't have proper task_id".format(task_id))
            raise RuntimeError(
                "invalid learner task_id({}) for finish task, valid learner_id is {}".format(
                    task_id, self._task_state.keys()
                )
            )
        with self._commander_lock:
            buffer_id = self._commander.finish_learner_task(task_id, finished_task)
        self.info("learner task({}) finish".format(task_id))
        # delete replay buffer
        if buffer_id is not None:
            replay_buffer = self._replay_buffer.pop(buffer_id)
            replay_buffer.close()
            self.info('replay_buffer({}) is closed'.format(buffer_id))

    def info(self, s: str) -> None:
        self._logger.info('[Coordinator({})]: {}'.format(self._coordinator_uid, s))

    def error(self, s: str) -> None:
        self._logger.error('[Coordinator({})]: {}'.format(self._coordinator_uid, s))

    def _record_task(self, task: dict):
        self._task_state[task['task_id']] = TaskState(task['task_id'])

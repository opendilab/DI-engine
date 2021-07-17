import time
import copy
from typing import List
from queue import Queue
from threading import Thread
from easydict import EasyDict

from ding.utils import build_logger, LockContext, LockContextType, get_task_uid
from ding.worker import create_buffer
from .comm_coordinator import CommCoordinator
from .base_parallel_commander import create_parallel_commander


class TaskState(object):
    r"""
    Overview:
        State recorder of the task, including ``task_id`` and ``start_time``.
    Interface:
        __init__
    """

    def __init__(self, task_id: str) -> None:
        r"""
        Overview:
            Init the task tate according to task_id and the init time.
        """
        self.task_id = task_id
        self.start_time = time.time()


class Coordinator(object):
    r"""
    Overview:
        the coordinator will manage parallel tasks and data
    Interface:
        __init__, start, close, __del__, state_dict, load_state_dict,
        deal_with_collector_send_data, deal_with_collector_finish_task,
        deal_with_learner_get_data, deal_with_learner_send_info, deal_with_learner_finish_task
    Property:
        system_shutdown_flag
    """
    config = dict(
        collector_task_timeout=30,
        learner_task_timeout=600,
        operator_server=dict(),
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        r"""
        Overview:
            init method of the coordinator
        Arguments:
            - cfg (:obj:`dict`): the config file to init the coordinator
        """
        self._exp_name = cfg.main.exp_name
        self._coordinator_uid = get_task_uid()
        coor_cfg = cfg.system.coordinator
        self._collector_task_timeout = coor_cfg.collector_task_timeout
        self._learner_task_timeout = coor_cfg.learner_task_timeout

        self._callback = {
            'deal_with_collector_send_data': self.deal_with_collector_send_data,
            'deal_with_collector_judge_finish': self.deal_with_collector_judge_finish,
            'deal_with_collector_finish_task': self.deal_with_collector_finish_task,
            'deal_with_learner_get_data': self.deal_with_learner_get_data,
            'deal_with_learner_send_info': self.deal_with_learner_send_info,
            'deal_with_learner_judge_finish': self.deal_with_learner_judge_finish,
            'deal_with_learner_finish_task': self.deal_with_learner_finish_task,
            'deal_with_increase_collector': self.deal_with_increase_collector,
            'deal_with_decrease_collector': self.deal_with_decrease_collector,
        }
        self._logger, _ = build_logger(path='./{}/log'.format(self._exp_name), name='coordinator', need_tb=False)
        self._interaction = CommCoordinator(coor_cfg, self._callback, self._logger)
        self._learner_task_queue = Queue()
        self._collector_task_queue = Queue()
        self._commander = create_parallel_commander(cfg.main)  # commander can access all the main config
        self._commander_lock = LockContext(LockContextType.THREAD_LOCK)
        # ############## Thread #####################
        # Assign thread todo
        # Produce thread todo
        self._assign_collector_thread = Thread(
            target=self._assign_collector_task, args=(), name='coordinator_assign_collector'
        )
        self._assign_learner_thread = Thread(
            target=self._assign_learner_task, args=(), name='coordinator_assign_learner'
        )
        self._produce_collector_thread = Thread(
            target=self._produce_collector_task, args=(), name='coordinator_produce_collector'
        )
        self._produce_learner_thread = Thread(
            target=self._produce_learner_task, args=(), name='coordinator_produce_learner'
        )

        self._replay_buffer = {}
        self._task_state = {}  # str -> TaskState
        self._historical_task = []
        # TODO remove used data
        # TODO load/save state_dict
        self._end_flag = True
        self._system_shutdown_flag = False

    def _assign_collector_task(self) -> None:
        r"""
        Overview:
            The function to be called in the assign_collector_task thread.
            Will get an collector task from ``collector_task_queue`` and assign the task.
        """
        while not self._end_flag:
            time.sleep(0.01)
            # get valid task, abandon timeout task
            if self._collector_task_queue.empty():
                continue
            else:
                collector_task, put_time = self._collector_task_queue.get()
                start_retry_time = time.time()
                max_retry_time = 0.3 * self._collector_task_timeout
                while True:
                    # timeout or assigned to collector
                    get_time = time.time()
                    if get_time - put_time >= self._collector_task_timeout:
                        self.info(
                            'collector task({}) timeout: [{}, {}, {}/{}]'.format(
                                collector_task['task_id'], get_time, put_time, get_time - put_time,
                                self._collector_task_timeout
                            )
                        )
                        with self._commander_lock:
                            self._commander.notify_fail_collector_task(collector_task)
                        break
                    buffer_id = collector_task['buffer_id']
                    if buffer_id in self._replay_buffer:
                        if self._interaction.send_collector_task(collector_task):
                            self._record_task(collector_task)
                            self.info(
                                "collector_task({}) is successful to be assigned".format(collector_task['task_id'])
                            )
                            break
                        else:
                            self.info("collector_task({}) is failed to be assigned".format(collector_task['task_id']))
                    else:
                        self.info(
                            "collector_task({}) can't find proper buffer_id({})".format(
                                collector_task['task_id'], buffer_id
                            )
                        )
                    if time.time() - start_retry_time >= max_retry_time:
                        # reput into queue
                        self._collector_task_queue.put([collector_task, put_time])
                        self.info("collector task({}) reput into queue".format(collector_task['task_id']))
                        break
                    time.sleep(3)

    def _assign_learner_task(self) -> None:
        r"""
        Overview:
            The function to be called in the assign_learner_task thread.
            Will take a learner task from learner_task_queue and assign the task.
        """
        while not self._end_flag:
            time.sleep(0.01)
            if self._learner_task_queue.empty():
                continue
            else:
                learner_task, put_time = self._learner_task_queue.get()
                start_retry_time = time.time()
                max_retry_time = 0.1 * self._learner_task_timeout
                while True:
                    # timeout or assigned to learner
                    get_time = time.time()
                    if get_time - put_time >= self._learner_task_timeout:
                        self.info(
                            'learner task({}) timeout: [{}, {}, {}/{}]'.format(
                                learner_task['task_id'], get_time, put_time, get_time - put_time,
                                self._learner_task_timeout
                            )
                        )
                        with self._commander_lock:
                            self._commander.notify_fail_learner_task(learner_task)
                        break
                    if self._interaction.send_learner_task(learner_task):
                        self._record_task(learner_task)
                        # create replay_buffer
                        buffer_id = learner_task['buffer_id']
                        if buffer_id not in self._replay_buffer:
                            replay_buffer_cfg = learner_task.pop('replay_buffer_cfg')
                            self._replay_buffer[buffer_id] = create_buffer(replay_buffer_cfg, exp_name=self._exp_name)
                            self._replay_buffer[buffer_id].start()
                            self.info("replay_buffer({}) is created".format(buffer_id))
                        self.info("learner_task({}) is successful to be assigned".format(learner_task['task_id']))
                        break
                    else:
                        self.info("learner_task({}) is failed to be assigned".format(learner_task['task_id']))
                    if time.time() - start_retry_time >= max_retry_time:
                        # reput into queue
                        self._learner_task_queue.put([learner_task, put_time])
                        self.info("learner task({}) reput into queue".format(learner_task['task_id']))
                        break
                    time.sleep(3)

    def _produce_collector_task(self) -> None:
        r"""
        Overview:
            The function to be called in the ``produce_collector_task`` thread.
            Will ask commander to produce a collector task, then put it into ``collector_task_queue``.
        """
        while not self._end_flag:
            time.sleep(0.01)
            with self._commander_lock:
                collector_task = self._commander.get_collector_task()
                if collector_task is None:
                    continue
            self.info("collector task({}) put into queue".format(collector_task['task_id']))
            self._collector_task_queue.put([collector_task, time.time()])

    def _produce_learner_task(self) -> None:
        r"""
        Overview:
            The function to be called in the produce_learner_task thread.
            Will produce a learner task and put it into the learner_task_queue.
        """
        while not self._end_flag:
            time.sleep(0.01)
            with self._commander_lock:
                learner_task = self._commander.get_learner_task()
                if learner_task is None:
                    continue
            self.info("learner task({}) put into queue".format(learner_task['task_id']))
            self._learner_task_queue.put([learner_task, time.time()])

    def state_dict(self) -> dict:
        r"""
        Overview:
            Return empty state_dict.
        """
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Overview:
            Pass when load state_dict.
        """
        pass

    def start(self) -> None:
        r"""
        Overview:
            Start the coordinator, including lunching the interaction thread and the collector learner threads.
        """
        self._end_flag = False
        self._interaction.start()
        self._produce_collector_thread.start()
        self._assign_collector_thread.start()
        self._produce_learner_thread.start()
        self._assign_learner_thread.start()

    def close(self) -> None:
        r"""
        Overview:
            Close the coordinator, including closing the interaction thread, the collector learner threads and the \
                buffers.
        """
        if self._end_flag:
            return
        self._end_flag = True
        time.sleep(1)
        self._produce_collector_thread.join()
        self._assign_collector_thread.join()
        self._produce_learner_thread.join()
        self._assign_learner_thread.join()
        self._interaction.close()
        # close replay buffer
        replay_buffer_keys = list(self._replay_buffer.keys())
        for k in replay_buffer_keys:
            v = self._replay_buffer.pop(k)
            v.close()
        self.info('coordinator is closed')

    def __del__(self) -> None:
        r"""
        Overview:
            __del__ method will close the coordinator.
        """
        self.close()

    def deal_with_collector_send_data(self, task_id: str, buffer_id: str, data_id: str, data: dict) -> None:
        r"""
        Overview:
            deal with the data send from collector
        Arguments:
            - task_id (:obj:`str`): the collector task_id
            - buffer_id (:obj:`str`): the buffer_id
            - data_id (:obj:`str`): the data_id
            - data (:obj:`str`): the data to dealt with
        """
        if task_id not in self._task_state:
            self.error('collector task({}) not in self._task_state when send data, throw it'.format(task_id))
            return
        if buffer_id not in self._replay_buffer:
            self.error(
                "collector task({}) data({}) doesn't have proper buffer_id({})".format(task_id, data_id, buffer_id)
            )
            return
        self._replay_buffer[buffer_id].push(data, -1)
        self.info('collector task({}) send data({})'.format(task_id, data_id))

    def deal_with_collector_judge_finish(self, task_id: str, data: dict) -> bool:
        if task_id not in self._task_state:
            self.error('collector task({}) not in self._task_state when send data, throw it'.format(task_id))
            return False
        with self._commander_lock:
            collector_finish_flag = self._commander.judge_collector_finish(task_id, data)
        if collector_finish_flag:
            self.info('collector task({}) is finished'.format(task_id))
        return collector_finish_flag

    def deal_with_collector_finish_task(self, task_id: str, finished_task: dict) -> None:
        r"""
        Overview:
            finish the collector task
        Arguments:
            - task_id (:obj:`str`): the collector task_id
            - finished_task (:obj:`dict`): the finished_task
        """
        if task_id not in self._task_state:
            self.error('collector task({}) not in self._task_state when finish, throw it'.format(task_id))
            return
        # finish_task
        with self._commander_lock:
            # commander will judge whether the whole system is converged and shoule be shutdowned
            self._system_shutdown_flag = self._commander.finish_collector_task(task_id, finished_task)
        self._task_state.pop(task_id)
        self._historical_task.append(task_id)
        self.info('collector task({}) is finished'.format(task_id))

    def deal_with_learner_get_data(self, task_id: str, buffer_id: str, batch_size: int,
                                   cur_learner_iter: int) -> List[dict]:
        r"""
        Overview:
            learner get the data from buffer
        Arguments:
            - task_id (:obj:`str`): the learner task_id
            - buffer_id (:obj:`str`): the buffer_id
            - batch_size (:obj:`int`): the batch_size to sample
            - cur_learn_iter (:obj:`int`): the current learner iter num
        """
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
        return self._replay_buffer[buffer_id].sample(batch_size, cur_learner_iter)

    def deal_with_learner_send_info(self, task_id: str, buffer_id: str, info: dict) -> None:
        r"""
        Overview:
            the learner send the info and update the priority in buffer
        Arguments:
            - task_id (:obj:`str`): the learner task id
            - buffer_id (:obj:`str`): the buffer_id of buffer to add info to
            - info (:obj:`dict`): the info to add
        """
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
        self._replay_buffer[buffer_id].update(info['priority_info'])
        with self._commander_lock:
            self._commander.update_learner_info(task_id, info)
        self.info("learner task({}) send info".format(task_id))

    def deal_with_learner_judge_finish(self, task_id: str, info: dict) -> bool:
        if task_id not in self._task_state:
            self.error("learner task({}) finish task doesn't have proper task_id".format(task_id))
            raise RuntimeError(
                "invalid learner task_id({}) for finish task, valid learner_id is {}".format(
                    task_id, self._task_state.keys()
                )
            )
        with self._commander_lock:
            learner_finish_flag = self._commander.judge_learner_finish(task_id, info)
        if learner_finish_flag:
            self.info('learner task({}) is finished'.format(task_id))
        return learner_finish_flag

    def deal_with_learner_finish_task(self, task_id: str, finished_task: dict) -> None:
        r"""
        Overview:
            finish the learner task, close the corresponding buffer
        Arguments:
            - task_id (:obj:`str`): the learner task_id
            - finished_task (:obj:`dict`): the dict of task to finish
        """
        if task_id not in self._task_state:
            self.error("learner task({}) finish task doesn't have proper task_id".format(task_id))
            raise RuntimeError(
                "invalid learner task_id({}) for finish task, valid learner_id is {}".format(
                    task_id, self._task_state.keys()
                )
            )
        with self._commander_lock:
            buffer_id = self._commander.finish_learner_task(task_id, finished_task)
        self._task_state.pop(task_id)
        self._historical_task.append(task_id)
        self.info("learner task({}) finish".format(task_id))
        # delete replay buffer
        if buffer_id is not None:
            replay_buffer = self._replay_buffer.pop(buffer_id)
            replay_buffer.close()
            self.info('replay_buffer({}) is closed'.format(buffer_id))

    def deal_with_increase_collector(self):
        r""""
        Overview:
        Increase task space when a new collector has added dynamically.
        """
        with self._commander_lock:
            self._commander.increase_collector_task_space()

    def deal_with_decrease_collector(self):
        r""""
        Overview:
        Decrease task space when a new collector has removed dynamically.
        """
        with self._commander_lock:
            self._commander.decrease_collector_task_space()

    def info(self, s: str) -> None:
        r"""
        Overview:
            Return the info
        Arguments:
            - s (:obj:`str`): the string to print in info
        """
        self._logger.info('[Coordinator({})]: {}'.format(self._coordinator_uid, s))

    def error(self, s: str) -> None:
        r"""
        Overview:
            Return the error
        Arguments:
            - s (:obj:`str`): the error info to print
        """
        self._logger.error('[Coordinator({})]: {}'.format(self._coordinator_uid, s))

    def _record_task(self, task: dict):
        r"""
        Overview:
            Create task state to record task
        Arguments:
            - task (:obj:`dict`): the task dict
        """
        self._task_state[task['task_id']] = TaskState(task['task_id'])

    @property
    def system_shutdown_flag(self) -> bool:
        r"""
        Overview:
            Return whether the system is shutdown
        Returns:
            - system_shutdown_flag (:obj:`bool`): whether the system is shutdown
        """
        return self._system_shutdown_flag

import os
import time
import traceback
from typing import Union, Dict, Callable
from queue import Queue
from threading import Thread

from nervex.utils import read_file, save_file
from nervex.interaction import Slave, TaskFail
from .base_comm_actor import BaseCommActor, register_comm_actor


class ActorSlave(Slave):
    """
    Overview:
        A slave, whose master is coordinator.
        Used to pass message between comm actor and coordinator.
    Interfaces:
        __init__, _process_task
    """

    # override
    def __init__(self, *args, callback_fn: Dict[str, Callable], **kwargs) -> None:
        """
        Overview:
            Init callback functions additionally. Callback functions are methods in comm actor.
        """
        super().__init__(*args, **kwargs)
        self._callback_fn = callback_fn
        self._current_task_info = None

    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
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
            return self._callback_fn['deal_with_resource']()
        elif task_name == 'actor_start_task':
            self._current_task_info = task['task_info']
            self._callback_fn['deal_with_actor_start'](self._current_task_info)
            return {'message': 'actor task has started'}
        elif task_name == 'actor_data_task':
            data = self._callback_fn['deal_with_actor_data']()
            data['buffer_id'] = self._current_task_info['buffer_id']
            data['task_id'] = self._current_task_info['task_id']
            return data
        elif task_name == 'actor_close_task':
            data = self._callback_fn['deal_with_actor_close']()
            data['task_id'] = self._current_task_info['task_id']
            return data
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))


class FlaskFileSystemActor(BaseCommActor):
    """
    Overview:
        An implementation of CommLearner, using flask and the file system.
    Interfaces:
        __init__, deal_with_resource, deal_with_actor_start, deal_with_actor_data, deal_with_actor_close,\
        get_policy_update_info, send_stepdata, send_metadata, start, close
    """

    # override
    def __init__(self, cfg: dict) -> None:
        """
       Overview:
            Initialization method.
       Arguments:
            - cfg (:obj:`EasyDict`): Config dict
       """
        BaseCommActor.__init__(self, cfg)
        host, port = cfg.host, cfg.port
        self._callback_fn = {
            'deal_with_resource': self.deal_with_resource,
            'deal_with_actor_start': self.deal_with_actor_start,
            'deal_with_actor_data': self.deal_with_actor_data,
            'deal_with_actor_close': self.deal_with_actor_close,
        }
        self._slave = ActorSlave(host, port, callback_fn=self._callback_fn)

        self._path_policy = cfg.path_policy
        self._path_data = cfg.path_data
        if not os.path.exists(self._path_data):
            try:
                os.mkdir(self._path_data)
            except Exception as e:
                pass
        self._metadata_queue = Queue(cfg.queue_maxsize)
        self._actor_close_flag = False
        self._actor = None

    def deal_with_resource(self) -> dict:
        """
        Overview:
            Callback function in ``ActorSlave``. Return how many resources are needed to start current actor.
        Returns:
            - resource (:obj:`dict`): Resource info dict, including ['gpu', 'cpu'].
        """
        return {'gpu': 1, 'cpu': 20}

    def deal_with_actor_start(self, task_info: dict) -> None:
        """
        Overview:
            Callback function in ``ActorSlave``. Create an actor and start an actor thread of the created one.
        Arguments:
            - task_info (:obj:`dict`): Task info dict.
        Note:
            In ``_create_actor`` method in base class ``BaseCommActor``, 4 methods
            'send_metadata', 'send_stepdata', 'get_policy_update_info', and policy are set.
            You can refer to it for details.
        """
        self._actor_close_flag = False
        self._actor = self._create_actor(task_info)
        self._actor_thread = Thread(target=self._actor.start, args=(), daemon=True, name='actor_start')
        self._actor_thread.start()

    def deal_with_actor_data(self) -> dict:
        """
        Overview:
            Callback function in ``ActorSlave``. Get data sample dict from ``_metadata_queue``,
            which will be sent to coordinator afterwards.
        Returns:
            - data (:obj:`Any`): Data sample dict.
        """
        while True:
            if not self._metadata_queue.empty():
                data = self._metadata_queue.get()
                break
            else:
                time.sleep(0.1)
        return data

    def deal_with_actor_close(self) -> dict:
        self._actor_close_flag = True
        finish_info = self._actor.get_finish_info()
        self._actor.close()
        self._actor_thread.join()
        del self._actor_thread
        self._actor = None
        return finish_info

    # override
    def get_policy_update_info(self, path: str) -> dict:
        """
        Overview:
            Get policy information in corresponding path.
        Arguments:
            - path (:obj:`str`): path to policy update information.
        """
        path = os.path.join(self._path_policy, path)
        return read_file(path, use_lock=True)

    # override
    def send_stepdata(self, path: str, stepdata: list) -> None:
        """
        Overview:
            Save actor's step data in corresponding path.
        Arguments:
            - path (:obj:`str`): Path to save data.
            - stepdata (:obj:`Any`): Data of one step.
        """
        if self._actor_close_flag:
            return
        name = os.path.join(self._path_data, path)
        save_file(name, stepdata, use_lock=False)

    # override
    def send_metadata(self, metadata: dict) -> None:
        """
        Overview:
            Store learn info dict in queue, which will be retrieved by callback function "deal_with_actor_learn"
            in actor slave, then will be sent to coordinator.
        Arguments:
            - metadata (:obj:`Any`): meta data.
        """
        if self._actor_close_flag:
            return
        necessary_metadata_keys = set(['data_id', 'policy_iter'])
        necessary_info_keys = set(['actor_done', 'cur_episode', 'cur_sample', 'cur_step'])
        assert necessary_metadata_keys.issubset(set(metadata.keys())
                                                ) or necessary_info_keys.issubset(set(metadata.keys()))
        while True:
            if not self._metadata_queue.full():
                self._metadata_queue.put(metadata)
                break
            else:
                time.sleep(0.1)

    def start(self) -> None:
        """
        Overview:
            Start comm actor itself and the actor slave.
        """
        BaseCommActor.start(self)
        self._slave.start()

    def close(self) -> None:
        """
        Overview:
            Close comm actor itself and the actor slave.
        """
        if self._end_flag:
            return
        if self._actor is not None:
            self._actor.close()
            self._actor_thread.join()
            del self._actor_thread
            self._actor = None
        self._slave.close()
        BaseCommActor.close(self)

    def __del__(self) -> None:
        self.close()


register_comm_actor('flask_fs', FlaskFileSystemActor)

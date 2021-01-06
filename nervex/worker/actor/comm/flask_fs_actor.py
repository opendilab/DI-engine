import os
import sys
import time
import traceback
from typing import Union
from queue import Queue
from threading import Thread

from nervex.utils import read_file, save_file
from nervex.interaction import Slave, TaskFail
from .base_comm_actor import BaseCommActor, register_comm_actor


class FlaskFileSystemActor(BaseCommActor, Slave):

    def __init__(self, cfg: dict) -> None:
        BaseCommActor.__init__(self, cfg)
        host, port = cfg.host, cfg.port
        Slave.__init__(self, host, port)
        self._job_request_id = 0

        self._path_policy = cfg.path_policy
        self._path_data = cfg.path_data
        if not os.path.exists(self._path_data):
            try:
                os.mkdir(self._path_data)
            except Exception as e:
                pass
        self._metadata_queue = Queue(cfg.queue_maxsize)
        self._finish_queue = Queue(cfg.queue_maxsize)
        self._actor = None

    # override Slave
    def _process_task(self, task: dict) -> Union[dict, TaskFail]:

        def run_actor():
            self._actor.start()

        task_name = task['name']
        if task_name == 'resource':
            return {'gpu': 1, 'cpu': 20}
        elif task_name == 'actor_start_task':
            self._current_task_info = task['task_info']
            self._actor = self._create_actor(self._current_task_info)
            self._actor_thread = Thread(target=run_actor, args=(), daemon=True)
            self._actor_thread.start()
            return {'message': 'actor task has started'}
        elif task_name == 'actor_data_task':
            while True:
                if not self._metadata_queue.empty():
                    data = self._metadata_queue.get()
                    break
                elif not self._finish_queue.empty():
                    data = self._finish_queue.get()
                    self._actor.close()
                    self._actor = None
                    break
                else:
                    time.sleep(0.1)
            data['buffer_id'] = self._current_task_info['buffer_id']
            data['task_id'] = self._current_task_info['task_id']
            return data
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))

    # override
    def get_policy_update_info(self, path: str) -> dict:
        path = os.path.join(self._path_policy, path)
        return read_file(path)

    # override
    def send_stepdata(self, path: str, stepdata: list) -> None:
        name = os.path.join(self._path_data, path)
        save_file(name, stepdata)

    # override
    def send_metadata(self, metadata: dict) -> None:
        necessary_metadata_keys = set(['data_id', 'policy_iter'])
        assert necessary_metadata_keys.issubset(set(metadata.keys()))
        while True:
            if not self._metadata_queue.full():
                self._metadata_queue.put(metadata)
                break
            else:
                time.sleep(0.1)

    # override
    def send_finish_info(self, finish_info: dict) -> None:
        necessary_finish_info_keys = set(['finished_task'])
        assert necessary_finish_info_keys.issubset(set(finish_info.keys()))
        while True:
            if not self._finish_queue.full():
                self._finish_queue.put(finish_info)
                break
            else:
                time.sleep(0.1)

    def start(self) -> None:
        BaseCommActor.start(self)
        Slave.start(self)

    def close(self) -> None:
        if self._end_flag:
            return
        if self._actor is not None:
            self._actor.close()
        Slave.close(self)
        BaseCommActor.close(self)

    def __del__(self) -> None:
        self.close()


register_comm_actor('flask_fs', FlaskFileSystemActor)

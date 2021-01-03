import os
import sys
import time
import traceback
from typing import Union
from queue import Queue

from nervex.utils import read_file, save_file
from nervex.interaction import Slave, TaskFail
from .base_comm_actor import BaseCommActor, register_comm_actor


class FlaskFileSystemActor(BaseCommActor, Slave):

    def __init__(self, cfg: dict) -> None:
        BaseCommActor.__init__(self, cfg)
        host, port = cfg.host, cfg.port
        Slave.__init__(self, host, port)
        self._job_request_id = 0

        self._path_agent = cfg.path_agent
        self._path_data = cfg.path_data
        self._metadata_queue = Queue(cfg._queue_maxsize)
        self._finish_queue = Queue(cfg._queue_maxsize)

    # override Slave
    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        task_name = task['name']
        if task_name == 'resource':
            return {'gpu': 1, 'cpu': 20}
        elif task_name == 'actor_state_task':
            self._current_task_info = task['task_info']
            self._actor = self._create_actor(self._current_task_info)
            # actor start
            return {'message': 'actor task has started'}
        elif task_name == 'actor_data_task':
            while True:
                if not self._metadata_queue.empty():
                    data = self._metadata_queue.get()
                    break
                elif not self._finish_queue.empty():
                    data = self._finish_queue.get()
                    break
                else:
                    time.sleep(0.1)
            data['buffer_id'] = self._current_task_info['buffer_id']
            data['task_id'] = self._current_task_info['task_id']
            return data
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))

    # override
    def get_agent_update_info(self, path: str) -> dict:
        path = os.path.join(self._path_agent, path)
        return read_file(path)

    # override
    def send_stepdata(self, path: str, stepdata: list) -> None:
        name = os.path.join(self._path_traj, path)
        save_file(name, stepdata)

    # override
    def send_metadata(self, metadata: dict) -> None:
        necessary_metadata_keys = set(['data_id', 'model_iter'])
        assert necessary_metadata_keys.is_subset(set(metadata.keys))
        while True:
            if not self._metadata_queue.full():
                self._metadata_queue.put(metadata)
                break
            else:
                time.sleep(0.1)

    # override
    def send_finish_job(self, finish_info: dict) -> None:
        necessary_finish_info_keys = set(['finished_task'])
        assert necessary_finish_info_keys.is_subset(set(finish_info.keys))
        while True:
            if not self._finish_queue.full():
                self._finish_queue.put(finish_info)
                break
            else:
                time.sleep(0.1)


register_comm_actor('flask_fs', FlaskFileSystemActor)

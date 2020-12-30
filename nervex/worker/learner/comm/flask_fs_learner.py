import os
import sys
import time
import traceback

import requests
from typing import List, Union
from functools import partial
from queue import Queue

from nervex.utils import read_file, save_file, get_rank, get_world_size, get_data_decompressor
from nervex.interaction import Slave, TaskFail
from .base_comm_learner import BaseCommLearner, register_comm_learner
from ..learner_hook import LearnerHook


class FlaskFileSystemLearner(BaseCommLearner, Slave):
    """
    Overview:
        An implementation of CommLearner, using flask as the file system.
    Interfaces:
        __init__, send_agent, get_data, send_learn_info, init_service, close_service,
    Property:
        hooks4call
    """

    def __init__(self, cfg: 'EasyDict') -> None:  # noqa
        """
        Overview:
            Initialize file path(url, path of traj & agent), comm frequency, dist learner info according to cfg.
        Arguments:
            - cfg (:obj:`EasyDict`): config dict
        """
        BaseCommLearner.__init__(self, cfg)
        host, port = cfg.upstream_ip, cfg.upstream_port
        Slave.__init__(self, host, port)

        self._path_traj = cfg.path_traj
        self._path_agent = cfg.path_agent
        self._send_agent_freq = cfg.send_agent_freq
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._restore = cfg.restore

        self._current_task_info = None
        self._data_demand_queue = Queue(maxsize=1)
        self._data_result_queue = Queue(maxsize=1)
        self._learn_info_queue = Queue(maxsize=1)

    # override Slave
    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        task_name = task['name']
        if task_name == 'resource':
            return {'gpu': self._world_size}
        elif task_name == 'learner_start_task':
            self._current_task_info = task['task_info']
            self._learner = self._create_learner(self._current_task_info)
            self._learner.start()
            return {'message': 'learner task has started'}
        elif task_name == 'learner_get_data_task':
            data_demand = self._data_demand_queue.get()
            return {
                'task_id': self._current_task_info['task_id'],
                'buffer_id': self._current_task_info['buffer_id'],
                'batch_size': data_demand
            }
        elif task_name == 'learner_learn_task':
            data = task['data']
            self._data_result_queue.put(data)
            learn_info = self._learn_info_queue.get()
            ret = {
                'info': learn_info,
                'task_id': self._current_task_info['task_id'],
                'buffer_id': self._current_task_info['buffer_id']
            }
            finished_task = learn_info.get('finished_task', None)
            if finished_task is not None:
                finished_task['buffer_id'] = self._current_task_info['buffer_id']
                self._current_task_info = None
                ret['finished_task'] = finished_task
                self._learner.close()
                self._learner = None
            else:
                ret['finished_task'] = None
            return ret
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))

    def init_service(self):
        BaseCommLearner.init_service(self)
        Slave.start(self)

    def close_service(self):
        BaseCommLearner.close_service(self)
        Slave.close(self)

    # override
    def send_agent(self, state_dict: dict) -> None:
        """
        Overview:
            Save learner's agent in corresponding path, called by ``SendAgentHook``.
        Arguments:
            - state_dict (:obj:`dict`): state dict of the runtime agent
        """
        path = os.path.join(self._path_agent, self._agent_name)
        save_file(path, state_dict)

    @staticmethod
    def load_data_fn(path, meta, decompressor):
        # due to read-write conflict, read_file may be error, therefore we circle this procedure
        while True:
            try:
                s = read_file(path)
                s = decompressor(s)
                break
            except Exception as e:
                time.sleep(0.01)
        begin, end = meta['unroll_split_begin'], meta['unroll_split_begin'] + meta['unroll_len']
        if meta['unroll_len'] == 1:
            s = s[begin]
            s.update(meta)
        else:
            s = s[begin:end]
            # add metdata key-value to stepdata
            for i in range(len(s)):
                s[i].update(meta)
        return s

    # override
    def get_data(self, batch_size: int) -> list:  # todo: doc not finished
        """
        Overview:
            Get batched data from coordinator.
        Arguments:
            - batch_size (:obj:`int`): size of one batch
        Returns:
            - data (:obj:`list`): a list of train data, each element is one traj
        """
        assert self._data_demand_queue.qsize() == 0
        sleep_count = 1
        while True:
            self._data_demand_queue.put(batch_size)
            data = self._data_result_queue.get()
            if data is not None:
                break
            time.sleep(sleep_count)
            sleep_count += 5
        assert isinstance(data, list)
        assert len(data) == batch_size, '{}/{}'.format(len(data), batch_size)
        decompressor = get_data_decompressor(data[0].get('compressor', 'none'))
        data = [
            partial(
                FlaskFileSystemLearner.load_data_fn,
                path=os.path.join(self._path_traj, m['data_id']),
                meta=m,
                decompressor=decompressor,
            ) for m in data
        ]
        return data

    # override
    def send_learn_info(self, learn_info: dict) -> None:
        """
        Overview:
            Send learn info to coordinator, called by ``SendLearnInfoHook``.
            Sending will repeat until succeeds or ``_active_flag`` is set to False.
        Arguments:
            - learn info (:obj:`dict`): learn info in `dict` type, \
                including keys `learn_info`(e.g. last iter, priority info)
        """
        assert self._learn_info_queue.qsize() == 0
        self._learn_info_queue.put(learn_info)

    @property
    def hooks4call(self) -> List[LearnerHook]:
        """
        Overview:
            Initialize the hooks and return them.
        Returns:
            - hooks (:obj:`list`): the hooks which comm learner have, will be registered in learner as well.
        """
        return [
            SendAgentHook('send_agent', 100, position='before_run', ext_args={}),
            SendAgentHook(
                'send_agent', 100, position='after_iter', ext_args={'send_agent_freq': self._send_agent_freq}
            ),
            SendLearnInfoHook(
                'send_learn_info',
                100,
                position='after_iter',
                ext_args={'freq': 10},
            ),
        ]


class SendAgentHook(LearnerHook):
    """
    Overview:
        Hook to send agent
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: dict = {}, **kwargs) -> None:
        """
        Overview:
            init SendAgentHook
        Arguments:
            - ext_args (:obj:`dict`): extended_args, use ext_args.freq to set send_agent_freq
        """
        super().__init__(*args, **kwargs)
        if 'send_agent_freq' in ext_args:
            self._freq = ext_args['send_agent_freq']
        else:
            self._freq = 1

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Save learner's agent in corresponding path at interval iterations, including model_state_dict, last_iter
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """
        last_iter = engine.last_iter.val
        if engine.rank == 0 and last_iter % self._freq == 0:
            state_dict = {'model': engine.agent.model.state_dict(), 'iter': last_iter}
            engine.send_agent(state_dict)
            engine.info('{} save iter{} agent'.format(engine.name, last_iter))


class SendLearnInfoHook(LearnerHook):
    """
    Overview:
        Hook to send learn info
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: dict, **kwargs) -> None:
        """
        Overview:
            init SendLearnInfoHook
        Arguments:
            - ext_args (:obj:`dict`): extended_args, use ext_args.freq
        """
        super().__init__(*args, **kwargs)
        self._freq = ext_args['freq']

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Send learn info including last_iter at interval iterations and priority info
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """
        last_iter = engine.last_iter.val
        engine.send_learn_info(engine.learn_info)
        if last_iter % self._freq == 0:
            engine.info('{} save iter{} learn_info'.format(engine.name, last_iter))


register_comm_learner('flask_fs', FlaskFileSystemLearner)
